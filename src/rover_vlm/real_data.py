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
from dataclasses import dataclass, fields
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
    backproject_depth,
    densify,
    drop_to_plane,
    fit_floor_plane,
    front_runs,
    interpolate_pose,
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

    def pose_at(self, t, max_gap_s):
        """Pose interpolated at exactly `t`, or None.

        Mocap/odometry samples rarely land on a frame timestamp, and `index_at`'s
        tolerance throws the frame away when none does -- even though TUM's mocap runs at
        300 Hz and the true pose is bracketed on both sides. Interpolates when `t` falls
        between two samples no more than `max_gap_s` apart; outside the trajectory, or
        across a real dropout, returns None.
        """
        if len(self.t) < 2 or t < self.t[0] or t > self.t[-1]:
            return None
        i = int(np.searchsorted(self.t, t))
        if i == 0:
            return self.T[0].copy()
        i0, i1 = i - 1, min(i, len(self.t) - 1)
        span = self.t[i1] - self.t[i0]
        if span > max_gap_s:
            return None
        u = 0.0 if span <= 0 else (t - self.t[i0]) / span
        return interpolate_pose(self.T[i0], self.T[i1], u)

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
        self._depth_t = d_t
        self._depth_paths = [self.root / v[0] for _, v in depth]
        self.frames = []
        for t, (fname,) in rgb:
            j = int(np.argmin(np.abs(d_t - t))) if len(d_t) else None
            dpath = self.root / depth[j][1][0] if j is not None and abs(d_t[j] - t) <= assoc_tol_s else None
            self.frames.append(Frame(t, self.root / fname, dpath))
        T = [pose_matrix(tuple(map(float, v[:3])), tuple(map(float, v[3:7]))) for _, v in gt]
        self.traj = Trajectory([t for t, _ in gt], T)  # camera optical centre in mocap world

    def summary(self):
        return (f"[tum] {self.root.name}: {len(self.frames)} frames, "
                f"{len(self.traj.t)} poses, hfov {self.K.hfov_deg:.1f} deg")

    def camera_pose(self, t):
        i = self.traj.index_at(t, self.pose_tol_s)
        return None if i is None else self.traj.T[i]

    def load_depth(self, frame):
        if frame.depth is None:
            return None
        return self.load_depth_file(frame.depth)

    @staticmethod
    def load_depth_file(path):
        return np.asarray(Image.open(path), dtype=np.float64) / TUM_DEPTH_SCALE

    def nearest_depth(self, t):
        """(path, |dt|) of the closest depth image to time `t`, regardless of the strict
        association tolerance used when the frames were built. TUM's colour and depth
        streams are both ~16 Hz but not locked, so a frame can miss the 20 ms association
        window while a perfectly usable depth image sits 40 ms away."""
        if not len(self._depth_t):
            return None, float("inf")
        j = int(np.argmin(np.abs(self._depth_t - t)))
        return self._depth_paths[j], float(abs(self._depth_t[j] - t))

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


def _goal_index_on_path(floor, K, max_goal_angle_deg, min_goal_dist_m, tail_m=0.0, rng=None):
    """Index into `floor` of the route point to use as the goal, or None.

    The training prompt asserts the goal is straight ahead, and the Habitat generator makes
    that exactly true (goal_x == 0.500 in every sample) by yawing the camera onto the goal.
    A real recording has no such guarantee: taking a fixed --horizon-m of driving and
    calling the endpoint "the goal" leaves it a median 37 degrees off axis, so the label
    contradicts the prompt and the goal metric scores that contradiction, not the model.

    So the goal is chosen from the route itself. A point qualifies when it is ahead of the
    camera, within `max_goal_angle_deg` of the optical axis, projects INSIDE the frame, and
    lies at least `min_goal_dist_m` along the track. Occlusion is deliberately not part of
    the test: Habitat's goals are on screen in 100 % of samples but unoccluded in only
    23 %, so a goal hidden behind an obstacle is the normal case, not a reject.

    Selection is by position along the route, never by euclidean distance -- the last
    qualifying point is the one furthest along the drive, which is not necessarily the one
    furthest away. Taking the LAST qualifying point rather than stopping at the first
    violation is what lets a route swerve around an obstacle and come back on axis with its
    bend intact; that is the Habitat case (A* detours, the goal stays centred) and the only
    kind of curved label both correct under the prompt and out of reach of a blind baseline.

    With `tail_m` > 0 the goal is drawn from `rng` uniformly over the qualifying points
    within that many metres of the last one, so the set gets a spread of goal ranges
    instead of every label ending at the same place. The tail is measured in metres rather
    than in samples because the sample rate differs by an order of magnitude between
    sources (TUM mocap ~300 Hz, GND odometry far slower), and "the last few samples" would
    otherwise mean centimetres on one and metres on the other.
    """
    seg = np.linalg.norm(np.diff(floor, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    uv, _z = project(floor, K)
    ang = np.degrees(np.arctan2(floor[:, 0], floor[:, 2]))
    ok = ((floor[:, 2] > 0) & (np.abs(ang) <= max_goal_angle_deg)
          & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
          & (uv[:, 0] >= 0) & (uv[:, 0] < K.width)
          & (uv[:, 1] >= 0) & (uv[:, 1] < K.height)
          & (cum >= min_goal_dist_m))
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return None
    if tail_m <= 0 or rng is None:
        return int(idx[-1])
    tail = idx[cum[idx] >= cum[idx[-1]] - tail_m]
    return int(rng.choice(tail.tolist()))


def build_real_record(rec_id, image_path, K: Intrinsics, T_w_c, future_w, plane: FloorPlane,
                      depth_m=None, meta=None, max_goal_angle_deg=45.0,
                      max_initial_angle_deg=30.0, step_m=0.05, allow_offscreen_goal=False,
                      goal_from_path=False, min_goal_dist_m=1.0, goal_tail_m=0.0, rng=None):
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
    goal_arc_m = None
    if goal_from_path:
        j = _goal_index_on_path(floor, K, max_goal_angle_deg, min_goal_dist_m,
                                tail_m=goal_tail_m, rng=rng)
        if j is None:
            return None, "no_on_axis_goal"
        floor = floor[:j + 1]
        goal_arc_m = float(np.linalg.norm(np.diff(floor, axis=0), axis=1).sum())
    end = floor[-1]
    if end[2] <= 0:
        return None, "endpoint_behind"
    goal_angle = float(np.degrees(np.arctan2(end[0], end[2])))
    init_angle = initial_heading_deg(floor)
    if not allow_offscreen_goal:
        if abs(goal_angle) > max_goal_angle_deg:
            return None, "endpoint_off_axis"
        if abs(init_angle) > max_initial_angle_deg:
            return None, "initial_off_axis"
    end_uv, _ = project(end[None, :], K)
    eu, ev = end_uv[0]
    if not (np.isfinite(eu) and np.isfinite(ev)):
        return None, "endpoint_out_of_frame"
    offscreen = not (0 <= eu < K.width and 0 <= ev < K.height)
    if offscreen and not allow_offscreen_goal:
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
        "goal_from_path": bool(goal_from_path),
        # distance ALONG the route, which is what --min-goal-dist-m bounds; goal_dist_m
        # above is the straight line to the same point, and is smaller whenever the
        # route curves -- a 3 m drive round a corner can end 1.4 m from where it started
        "goal_arc_m": None if goal_arc_m is None else round(goal_arc_m, 3),
        "frac_hidden": round(float(hidden.mean()), 3),
        "cam_height_m": round(plane.height, 3),
        "plane_tilt_deg": round(plane.tilt_deg, 2),
        "plane_inlier_frac": round(plane.inlier_frac, 3),
        "hfov_deg": round(K.hfov_deg, 2),
        "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy, "W": K.width, "H": K.height,
    }
    if allow_offscreen_goal:
        # the answer's goal is clamped to the frame; keep where it really projected, so a
        # "drive toward a destination you cannot see" case stays identifiable downstream
        info["goal_offscreen"] = bool(offscreen)
        info["goal_uv_norm"] = [round(float(eu / K.width), 3), round(float(ev / K.height), 3)]
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


def track_length(traj: Trajectory, i0, i1):
    """Distance travelled along the trajectory between two pose indices."""
    pos = traj.positions[i0:i1 + 1]
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0


def window_end_index(traj: Trajectory, i0, length_m, max_window_s, max_gap_s,
                     min_length_m=None):
    """Index of the pose where the path from i0 reaches length_m, or a skip reason string
    ('track_too_short' if the trajectory ends first, 'too_slow' if the window takes longer
    than max_window_s, 'gt_gap' if consecutive poses are more than max_gap_s apart).

    With `min_length_m`, a track that ends before reaching `length_m` is kept as long as
    it runs at least that far -- the last metres of every recording are otherwise thrown
    away for asking a fixed horizon of a finite drive.
    """
    pos = traj.positions[i0:]
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    hit = np.nonzero(cum >= length_m)[0]
    if not len(hit):
        if min_length_m is None or cum[-1] < min_length_m:
            return "track_too_short"
        hit = [len(cum) - 1]
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

    @staticmethod
    def _typestore(reader):
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore

        ts = get_typestore(Stores.ROS1_NOETIC)
        for c in reader.connections:
            if c.msgtype not in ts.types:
                ts.register(get_types_from_msg(c.msgdef.data, c.msgtype))
        return ts

    @classmethod
    def read_tf_static(cls, bag_path):
        """The latched /tf_static tree of one bag, as child -> (parent, T_parent_child).

        Only chunk01 of a GND recording carries /tf_static, but the transform is *static*
        for the whole recording -- the ZED does not move on its mount between chunks -- so
        chunk01's tree is the right one for chunk02..NN. That turns the other 170 chunk
        bags in the Dataverse from unusable into usable.
        """
        from rosbags.rosbag1 import Reader

        static = {}
        with Reader(Path(bag_path)) as r:
            ts = cls._typestore(r)
            conns = [c for c in r.connections if c.topic == "/tf_static"]
            if not conns:
                raise RuntimeError(f"{Path(bag_path).name}: no /tf_static to read")
            for con, _, raw in r.messages(connections=conns):
                m = ts.deserialize_ros1(raw, con.msgtype)
                for tr in m.transforms:
                    tt, q = tr.transform.translation, tr.transform.rotation
                    static[tr.child_frame_id] = (
                        tr.header.frame_id,
                        pose_matrix((tt.x, tt.y, tt.z), (q.x, q.y, q.z, q.w)))
        return static

    def __init__(self, bag_path, images_dir, cam_height, pose_tol_s=0.05, imu_window_s=0.25,
                 tf_static_from=None):
        from rosbags.rosbag1 import Reader

        if cam_height is None:
            raise ValueError("GND needs --cam-height: the bag's tf_static has no camera height")
        self.bag_path = Path(bag_path)
        self.cam_height = float(cam_height)
        self.pose_tol_s = pose_tol_s
        self.imu_window_s = imu_window_s
        out = Path(images_dir) / self.bag_path.stem
        out.mkdir(parents=True, exist_ok=True)

        frames, odom_t, odom_T, imu_t, imu_a, static = [], [], [], [], [], {}
        self.K = None
        with Reader(self.bag_path) as r:
            ts = self._typestore(r)
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
        self.tf_static_source = self.bag_path.name
        if not static and tf_static_from is not None:
            static = self.read_tf_static(tf_static_from)
            self.tf_static_source = Path(tf_static_from).name
        if not static:
            raise RuntimeError(
                f"{self.bag_path.name}: no /tf_static. Only chunk01 of a GND recording has it; "
                f"pass tf_static_from=<chunk01 of the same recording> to borrow it")
        self.frames = sorted(frames, key=lambda f: f.t)
        order = np.argsort(odom_t)
        self.traj = Trajectory(np.array(odom_t)[order], np.array(odom_T)[order])
        self.T_base_cam = _compose_chain(static, GND_BASE_FRAME, GND_CAMERA_FRAME)
        borrowed = "" if self.tf_static_source == self.bag_path.name else f"({self.tf_static_source})"
        self.extrinsic_source = f"tf_static{borrowed}+imu" if imu_t else f"tf_static{borrowed}"
        self.imu_t = np.array(imu_t)
        self.imu_a = np.array(imu_a).reshape(-1, 3)
        self.R_cam_imu = (_compose_chain(static, GND_CAMERA_FRAME, self.imu_frame)[:3, :3]
                          if imu_t else None)

    def summary(self):
        return (f"[gnd] {self.bag_path.name}: {len(self.frames)} frames, "
                f"{len(self.traj.t)} odom poses, K fx={self.K.fx:.1f} "
                f"hfov={self.K.hfov_deg:.1f} deg, floor normal from "
                f"{self.extrinsic_source}, cam height {self.cam_height:.3f} m (assumed)")

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


# --- per-frame walk --------------------------------------------------------------------

@dataclass(frozen=True)
class WindowParams:
    """Which trajectory window becomes a label, and which windows are rejected."""

    horizon_m: float | None = None   # fixed path length; None -> uniform in [min_len, max_len]
    min_len: float = 3.0
    max_len: float = 12.0
    stride_s: float = 1.0            # min time between chosen frames; 0 -> every frame
    max_window_s: float = 40.0
    max_gap_s: float = 0.5
    max_goal_angle: float = 45.0
    max_initial_angle: float = 30.0
    seed: int = 0
    # --- label recovery, all off by default so existing sets rebuild byte-identically ---
    depth_tol_s: float = 0.0     # >0: use the nearest depth image within this many seconds
    interp_pose: bool = False    # interpolate the trajectory at the frame time
    min_horizon_m: float | None = None  # accept a short window rather than dropping it
    allow_offscreen_goal: bool = False  # keep frames whose goal leaves the image
    # --- goal selection ---
    goal_from_path: bool = False   # goal = farthest on-axis point of the route, not
                                   # the fixed-horizon endpoint (see _goal_index_on_path)
    min_goal_dist_m: float = 1.0   # with goal_from_path, reject a goal nearer than this
    goal_tail_m: float = 0.0       # >0: draw the goal from the last this-many metres of
                                   # qualifying route instead of always taking the last point

    @classmethod
    def from_args(cls, args):
        """Pick the matching fields out of an argparse Namespace."""
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)})


@dataclass(frozen=True)
class FrameResult:
    """One visited frame. Exactly one of `record` / `reason` is set.

    `meta` is what was handed to `build_real_record` -- partial when the frame was rejected
    before the builder ran, but always carrying whatever was known at that point, so a
    caller rendering the frame can still say which horizon and which plane it was judging.
    """

    frame: Frame
    index: int
    record: dict | None
    reason: str | None
    meta: dict


def iter_tum_frames(seq: TumSequence, params: WindowParams, rng):
    """Yield a FrameResult per TUM frame at least `params.stride_s` after the last one.

    Rejected frames are yielded with their reason rather than dropped; the caller decides
    whether to count them (scripts/prepare_real_eval.py) or draw them
    (scripts/render_real_video.py).
    """
    last_plane = None
    next_t = -np.inf
    for k, frame in enumerate(seq.frames):
        if frame.t < next_t:
            continue
        next_t = frame.t + params.stride_s
        meta = {"dataset": "tum", "sequence": seq.root.name, "timestamp": frame.t}
        T_w_c = seq.camera_pose(frame.t)
        if T_w_c is None and params.interp_pose:
            T_w_c = seq.traj.pose_at(frame.t, params.max_gap_s)
            if T_w_c is not None:
                meta["pose_source"] = "interpolated"
        if T_w_c is None:
            yield FrameResult(frame, k, None, "no_gt", meta)
            continue
        L = params.horizon_m if params.horizon_m else rng.uniform(params.min_len, params.max_len)
        meta["horizon_m"] = round(L, 2)
        i0 = seq.traj.index_at(frame.t, seq.pose_tol_s)
        if i0 is None:  # interpolated pose: start the future track at the next sample
            i0 = int(np.searchsorted(seq.traj.t, frame.t))
        i1 = window_end_index(seq.traj, i0, L, params.max_window_s, params.max_gap_s,
                              min_length_m=params.min_horizon_m)
        if isinstance(i1, str):
            yield FrameResult(frame, k, None, i1, meta)
            continue
        if params.min_horizon_m is not None:
            meta["horizon_actual_m"] = round(track_length(seq.traj, i0, i1), 2)
        depth, depth_path = seq.load_depth(frame), frame.depth
        if depth is None and params.depth_tol_s > 0:
            cand, dt = seq.nearest_depth(frame.t)
            if cand is not None and dt <= params.depth_tol_s:
                depth, depth_path = seq.load_depth_file(cand), cand
                meta["depth_dt_s"] = round(dt, 4)
        if params.depth_tol_s > 0:
            meta["depth_source"] = ("frame" if depth_path is frame.depth and depth is not None
                                    else "nearest" if depth is not None else "none")
        elif depth is None:
            yield FrameResult(frame, k, None, "no_depth", meta)
            continue
        plane = (fit_floor_plane(backproject_depth(depth, seq.K), seed=params.seed + k)
                 if depth is not None else None)
        plane_source = "depth"
        if plane is None:
            if last_plane is None:
                yield FrameResult(frame, k, None, "plane_fit_failed", meta)
                continue
            plane, plane_source = last_plane, "previous_frame"
        else:
            last_plane = plane
        meta["plane_source"] = plane_source
        meta["gt_tz_m"] = round(float(T_w_c[2, 3]), 3)
        rec, reason = build_real_record(
            f"{seq.root.name}_{frame.image.stem}", frame.image, seq.K, T_w_c,
            seq.traj.positions[i0:i1 + 1], plane, depth_m=depth,
            max_goal_angle_deg=params.max_goal_angle,
            max_initial_angle_deg=params.max_initial_angle, meta=meta,
            allow_offscreen_goal=params.allow_offscreen_goal,
            goal_from_path=params.goal_from_path, min_goal_dist_m=params.min_goal_dist_m,
            goal_tail_m=params.goal_tail_m, rng=rng)
        yield FrameResult(frame, k, rec, reason, meta)


def iter_gnd_frames(bag, params: WindowParams, rng):
    """Yield a FrameResult per GND frame. No depth, so nothing is ever occluded."""
    next_t = -np.inf
    for k, frame in enumerate(bag.frames):
        if frame.t < next_t:
            continue
        next_t = frame.t + params.stride_s
        meta = {"dataset": "gnd", "sequence": bag.bag_path.stem, "timestamp": frame.t}
        T_w_c = bag.camera_pose(frame.t)
        if T_w_c is None and params.interp_pose:
            base = bag.traj.pose_at(frame.t, params.max_gap_s)
            if base is not None:
                T_w_c = base @ bag.T_base_cam
                meta["pose_source"] = "interpolated"
        if T_w_c is None:
            yield FrameResult(frame, k, None, "no_odom", meta)
            continue
        L = params.horizon_m if params.horizon_m else rng.uniform(params.min_len, params.max_len)
        meta["horizon_m"] = round(L, 2)
        i0 = bag.traj.index_at(frame.t, bag.pose_tol_s)
        if i0 is None:
            i0 = int(np.searchsorted(bag.traj.t, frame.t))
        i1 = window_end_index(bag.traj, i0, L, params.max_window_s, params.max_gap_s,
                              min_length_m=params.min_horizon_m)
        if isinstance(i1, str):
            yield FrameResult(frame, k, None, i1, meta)
            continue
        if params.min_horizon_m is not None:
            meta["horizon_actual_m"] = round(track_length(bag.traj, i0, i1), 2)
        meta["plane_source"] = bag.extrinsic_source
        rec, reason = build_real_record(
            f"{bag.bag_path.stem}_{frame.image.stem}", bag.materialize(frame), bag.K, T_w_c,
            bag.ground_positions(i0, i1), bag.floor_plane(frame.t), depth_m=None,
            max_goal_angle_deg=params.max_goal_angle,
            max_initial_angle_deg=params.max_initial_angle, meta=meta,
            allow_offscreen_goal=params.allow_offscreen_goal,
            goal_from_path=params.goal_from_path, min_goal_dist_m=params.min_goal_dist_m,
            goal_tail_m=params.goal_tail_m, rng=rng)
        yield FrameResult(frame, k, rec, reason, meta)
