"""Real-image eval records from robot video with a known trajectory.

For a frame at time t, the robot's *future* positions (until the path is L metres long)
are expressed in that frame's camera coordinates, dropped onto the floor plane, projected
into the image and occlusion-tested against the depth image. The resulting polyline then
goes through the SAME clip / resample / format functions as the Habitat training data
(`rover_vlm.habitat_data`), so the answer string is indistinguishable in form from a
Habitat label: {"path":[[x,y,v],...],"goal":[x,y,v]} with normalized coords.

Dataset adapters expose the few things the builder needs -- frames, camera intrinsics,
per-time camera pose, optional depth -- and nothing else:
  * `TumSequence`   TUM RGB-D freiburg2 pioneer sequences (mocap pose of the colour
                    camera's optical centre; registered Kinect depth for the floor plane
                    and visibility).
  * `GndBag`        GND rosbag (ZED2 rectified colour + EKF odometry of base_link; camera
                    extrinsic from tf_static; no depth -> flat ground, everything visible).

Camera convention throughout: OpenCV (+x right, +y down, +z forward). Records carry a
`real_meta` block that `scripts/evaluate.py` ignores.
"""

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rover_vlm.habitat_data import (
    HABITAT_PROMPT,
    _clamp01,
    clip_polyline_unit,
    format_answer,
    normalize_points,
    resample_with_transitions,
)
from rover_vlm.projection import (
    FloorPlane,
    Intrinsics,
    densify,
    drop_to_plane,
    front_runs,
    invert_pose,
    occluded_mask,
    pose_matrix,
    project,
    transform_points,
)

# TUM fr2 colour camera (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
TUM_FR2_K = Intrinsics(fx=520.9, fy=521.0, cx=325.1, cy=249.7, width=640, height=480)
TUM_DEPTH_SCALE = 5000.0  # png value per metre


@dataclass(frozen=True)
class Frame:
    t: float
    image: Path
    depth: Path | None = None


class Trajectory:
    """Time-stamped world poses of one frame (camera or base). Poses are 4x4 local->world."""

    def __init__(self, t, T):
        self.t = np.asarray(t, dtype=float)
        self.T = np.asarray(T, dtype=float)
        assert self.t.ndim == 1 and self.T.shape == (len(self.t), 4, 4)
        assert np.all(np.diff(self.t) >= 0), "trajectory must be time-sorted"

    @property
    def positions(self):
        return self.T[:, :3, 3]

    def index_at(self, t, tol_s):
        """Index of the nearest sample within tol_s of t, else None."""
        i = bisect.bisect_left(self.t, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - t) <= tol_s:
                if best is None or abs(self.t[j] - t) < abs(self.t[best] - t):
                    best = j
        return best


# --- TUM RGB-D -------------------------------------------------------------------------

def _read_tum_list(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), parts[1:]))
    return rows


class TumSequence:
    """One `rgbd_dataset_freiburg2_*` directory: rgb.txt / depth.txt / groundtruth.txt."""

    name = "tum"

    def __init__(self, root, assoc_tol_s=0.02, pose_tol_s=0.02):
        self.root = Path(root)
        self.K = TUM_FR2_K
        self.pose_tol_s = pose_tol_s
        rgb = _read_tum_list(self.root / "rgb.txt")
        depth = _read_tum_list(self.root / "depth.txt")
        gt = _read_tum_list(self.root / "groundtruth.txt")
        d_t = np.array([t for t, _ in depth])
        self.frames = []
        for t, (fname,) in rgb:
            j = int(np.argmin(np.abs(d_t - t))) if len(d_t) else None
            dpath = self.root / depth[j][1][0] if j is not None and abs(d_t[j] - t) <= assoc_tol_s else None
            self.frames.append(Frame(t, self.root / fname, dpath))
        T = [pose_matrix(tuple(map(float, v[:3])), tuple(map(float, v[3:7]))) for _, v in gt]
        self.traj = Trajectory([t for t, _ in gt], T)  # camera optical centre in mocap world

    def camera_pose(self, t):
        i = self.traj.index_at(t, self.pose_tol_s)
        return None if i is None else self.traj.T[i]

    def load_depth(self, frame):
        if frame.depth is None:
            return None
        return np.asarray(Image.open(frame.depth), dtype=np.float64) / TUM_DEPTH_SCALE

    def future_camera_positions(self, t, until_t):
        """World positions of the camera centre for t <= time <= until_t plus their times."""
        i0 = bisect.bisect_left(self.traj.t, t)
        i1 = bisect.bisect_right(self.traj.t, until_t)
        return self.traj.t[i0:i1], self.traj.positions[i0:i1]


# --- generic record builder ------------------------------------------------------------

def initial_heading_deg(floor, lead_m=1.0):
    """Bearing (deg, + = right) of the point `lead_m` along the floor track from its start."""
    seg = np.linalg.norm(np.diff(floor, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    j = int(np.searchsorted(cum, min(lead_m, cum[-1])))
    d = floor[j] - floor[0]
    return float(np.degrees(np.arctan2(d[0], d[2])))


def build_real_record(rec_id, image_path, K: Intrinsics, T_w_c, future_w, plane: FloorPlane,
                      depth_m=None, meta=None, max_goal_angle_deg=45.0,
                      max_initial_angle_deg=30.0, step_m=0.05):
    """Future world positions -> Habitat-format record, or (None, reason).

    `future_w` (M,3): the robot/camera track from the current time forward; its last
    point is the goal. Points are dropped onto `plane` in the camera frame, densified,
    projected, occlusion-tested against `depth_m` (metres, camera-aligned) when given,
    then clipped/resampled exactly like the Habitat labels.

    Two heading filters keep the label distribution Habitat-like (path starts at the
    rover and leads to a goal ahead): the first metre of travel must be within
    `max_initial_angle_deg` of the optical axis (the robot is driving forward, not
    spinning), and the goal within `max_goal_angle_deg`.
    """
    future_w = np.asarray(future_w, dtype=float).reshape(-1, 3)
    if len(future_w) < 2:
        return None, "track_too_short"
    p_c = transform_points(invert_pose(T_w_c), future_w)
    floor = drop_to_plane(p_c, plane)
    end = floor[-1]
    if end[2] <= 0:
        return None, "endpoint_behind"
    goal_angle = float(np.degrees(np.arctan2(end[0], end[2])))
    if abs(goal_angle) > max_goal_angle_deg:
        return None, "endpoint_off_axis"
    init_angle = initial_heading_deg(floor)
    if abs(init_angle) > max_initial_angle_deg:
        return None, "initial_off_axis"
    end_uv, _ = project(end[None, :], K)
    eu, ev = end_uv[0]
    if not (np.isfinite(eu) and np.isfinite(ev) and 0 <= eu < K.width and 0 <= ev < K.height):
        return None, "endpoint_out_of_frame"

    dense = densify(floor, step_m)
    uv, z = project(dense, K)
    runs = front_runs(z)
    if not runs:
        return None, "all_behind"
    idx = np.concatenate([np.arange(a, b) for a, b in runs])
    hidden = (occluded_mask(uv[idx], z[idx], depth_m) if depth_m is not None
              else np.zeros(len(idx), dtype=bool))
    raw = [(float(uv[i, 0]), float(uv[i, 1]), bool(h)) for i, h in zip(idx, hidden)]
    clipped = clip_polyline_unit(normalize_points(raw, K.width, K.height))
    if len(clipped) < 2:
        return None, "too_few_points"
    path_wps = resample_with_transitions(clipped)
    goal_wp = (_clamp01(eu / K.width), _clamp01(ev / K.height), 0 if raw[-1][2] else 1)
    info = {
        "goal_angle_deg": round(goal_angle, 2),
        "initial_angle_deg": round(init_angle, 2),
        "goal_dist_m": round(float(np.linalg.norm(end)), 3),
        "frac_hidden": round(float(hidden.mean()), 3),
        "cam_height_m": round(plane.height, 3),
        "plane_tilt_deg": round(plane.tilt_deg, 2),
        "plane_inlier_frac": round(plane.inlier_frac, 3),
        "hfov_deg": round(K.hfov_deg, 2),
        "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy, "W": K.width, "H": K.height,
    }
    if meta:
        info.update(meta)
    return {
        "id": rec_id,
        "image": [str(Path(image_path).resolve())],
        "conversations": [
            {"from": "human", "value": HABITAT_PROMPT},
            {"from": "gpt", "value": format_answer(path_wps, goal_wp)},
        ],
        "real_meta": info,
    }, None


def load_records(path):
    return json.loads(Path(path).read_text())


def window_end_index(traj: Trajectory, i0, length_m, max_window_s, max_gap_s):
    """Index of the pose where the path from i0 reaches length_m, or a skip reason string
    ('track_too_short' if the trajectory ends first, 'too_slow' if the window takes longer
    than max_window_s, 'gt_gap' if consecutive poses are more than max_gap_s apart)."""
    pos = traj.positions[i0:]
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    hit = np.nonzero(cum >= length_m)[0]
    if not len(hit):
        return "track_too_short"
    i1 = i0 + int(hit[0])
    if traj.t[i1] - traj.t[i0] > max_window_s:
        return "too_slow"
    if i1 > i0 and np.max(np.diff(traj.t[i0:i1 + 1])) > max_gap_s:
        return "gt_gap"
    return i1


# --- GND rosbag ------------------------------------------------------------------------

GND_IMAGE_TOPIC = "zed_node/rgb/image_rect_color/compressed"
GND_INFO_TOPIC = "zed_node/rgb/camera_info"
GND_ODOM_TOPIC = "/odometry/filtered"
GND_IMU_TOPIC = "zed_node/imu/data"
GND_CAMERA_FRAME = "zed2_left_camera_optical_frame"
GND_BASE_FRAME = "base_link"


def _to_root(static, frame):
    """T_root_frame by walking tf_static (child -> (parent, T_parent_child)) up to the root."""
    T = np.eye(4)
    cur = frame
    while cur in static:
        p, Tpc = static[cur]
        T = Tpc @ T
        cur = p
    return cur, T


def _compose_chain(static, a, b):
    """T_a_b between any two frames of one tf_static tree (via their common root)."""
    ra, Ta = _to_root(static, a)
    rb, Tb = _to_root(static, b)
    if ra != rb:
        raise KeyError(f"tf_static: {a} (root {ra}) and {b} (root {rb}) are not connected")
    return invert_pose(Ta) @ Tb


class GndBag:
    """One GND rosbag chunk. Reads the bag once: JPEG frames are written verbatim to
    `images_dir/<bag stem>/<stamp>.jpg`, odometry becomes the base_link trajectory, the
    tf_static chain gives T_base_cam, and the ZED IMU provides gravity for the floor
    normal. The camera height above ground is NOT in the bag (tf_static mounts the ZED at
    base_link) and must be passed in.
    """

    name = "gnd"

    def __init__(self, bag_path, images_dir, cam_height, pose_tol_s=0.05, imu_window_s=0.25):
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore

        if cam_height is None:
            raise ValueError("GND needs --cam-height: the bag's tf_static has no camera height")
        self.bag_path = Path(bag_path)
        self.cam_height = float(cam_height)
        self.pose_tol_s = pose_tol_s
        self.imu_window_s = imu_window_s
        out = Path(images_dir) / self.bag_path.stem
        out.mkdir(parents=True, exist_ok=True)

        ts = get_typestore(Stores.ROS1_NOETIC)
        frames, odom_t, odom_T, imu_t, imu_a, static = [], [], [], [], [], {}
        self.K = None
        with Reader(self.bag_path) as r:
            for c in r.connections:
                if c.msgtype not in ts.types:
                    ts.register(get_types_from_msg(c.msgdef.data, c.msgtype))
            conns = {c.topic: c for c in r.connections}
            wanted = [conns[t] for t in (GND_IMAGE_TOPIC, GND_INFO_TOPIC, GND_ODOM_TOPIC,
                                         GND_IMU_TOPIC, "/tf_static") if t in conns]
            for con, t_ns, raw in r.messages(connections=wanted):
                m = ts.deserialize_ros1(raw, con.msgtype)
                if con.topic == GND_IMAGE_TOPIC:
                    stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                    path = out / f"{stamp:.6f}.jpg"
                    if not path.exists():
                        path.write_bytes(bytes(m.data))
                    frames.append(Frame(stamp, path))
                elif con.topic == GND_INFO_TOPIC:
                    if self.K is None:
                        k = [float(x) for x in m.K]
                        self.K = Intrinsics(k[0], k[4], k[2], k[5], int(m.width), int(m.height))
                        assert all(float(x) == 0.0 for x in m.D), "expected a rectified stream"
                elif con.topic == GND_ODOM_TOPIC:
                    p, q = m.pose.pose.position, m.pose.pose.orientation
                    odom_t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                    odom_T.append(pose_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
                elif con.topic == GND_IMU_TOPIC:
                    a = m.linear_acceleration
                    imu_t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                    imu_a.append((a.x, a.y, a.z))
                    self.imu_frame = m.header.frame_id
                else:  # /tf_static
                    for tr in m.transforms:
                        tt, q = tr.transform.translation, tr.transform.rotation
                        static[tr.child_frame_id] = (
                            tr.header.frame_id,
                            pose_matrix((tt.x, tt.y, tt.z), (q.x, q.y, q.z, q.w)))
        if self.K is None:
            raise RuntimeError(f"{self.bag_path.name}: no {GND_INFO_TOPIC}")
        if not static:
            raise RuntimeError(f"{self.bag_path.name}: no /tf_static (use the first chunk of a recording)")
        self.frames = sorted(frames, key=lambda f: f.t)
        order = np.argsort(odom_t)
        self.traj = Trajectory(np.array(odom_t)[order], np.array(odom_T)[order])
        self.T_base_cam = _compose_chain(static, GND_BASE_FRAME, GND_CAMERA_FRAME)
        self.extrinsic_source = "tf_static+imu" if imu_t else "tf_static"
        self.imu_t = np.array(imu_t)
        self.imu_a = np.array(imu_a).reshape(-1, 3)
        self.R_cam_imu = (_compose_chain(static, GND_CAMERA_FRAME, self.imu_frame)[:3, :3]
                          if imu_t else None)

    def camera_pose(self, t):
        i = self.traj.index_at(t, self.pose_tol_s)
        return None if i is None else self.traj.T[i] @ self.T_base_cam

    def ground_positions(self, i0, i1):
        """World positions of base_link over [i0, i1], flattened to the current height
        (flat-ground assumption; the EKF altitude drifts and is not trusted)."""
        p = self.traj.positions[i0:i1 + 1].copy()
        p[:, 2] = p[0, 2]
        return p

    def floor_plane(self, t):
        """Floor plane in the camera frame at time t: normal from the IMU's mean
        acceleration (gravity, up) in a +-imu_window_s window, height from --cam-height.
        Falls back to the tf_static pitch when no IMU sample is near t."""
        if self.R_cam_imu is not None and len(self.imu_t):
            sel = np.abs(self.imu_t - t) <= self.imu_window_s
            if sel.sum() >= 5:
                up_imu = self.imu_a[sel].mean(axis=0)
                if np.linalg.norm(up_imu) > 5.0:  # sane gravity magnitude
                    n = self.R_cam_imu @ (up_imu / np.linalg.norm(up_imu))
                    return FloorPlane(tuple(float(x) for x in n), self.cam_height, 1.0)
        # tf_static only: floor normal = base +z expressed in the camera frame
        n = self.T_base_cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        return FloorPlane(tuple(float(x) for x in n), self.cam_height, 1.0)

    @staticmethod
    def materialize(frame):
        return frame.image
