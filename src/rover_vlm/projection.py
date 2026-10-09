"""Camera geometry for turning a real robot trajectory into image-space path labels.

Used by `rover_vlm.real_data` to build the real-image eval set: future robot positions
are expressed in the camera frame at time t, dropped onto the floor plane, projected
with the pinhole model, and tested for occlusion against the depth image. Mirrors the
primitives the Habitat generator used (`roverbench/perception/depth.py`, `viz.py`) so
the real labels follow the same rules as the synthetic ones (in-front runs, 0.15 m
occlusion margin), but in the **OpenCV camera convention**: +x right, +y down, +z forward.

Pure numpy, CPU only, no dataset-specific code.
"""

from dataclasses import dataclass

import numpy as np

Z_NEAR = 0.05  # m: points closer than this along the optical axis are "behind" the camera
OCCLUSION_MARGIN_M = 0.15  # Habitat's _occluded_mask margin; keeps the floor from self-occluding


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width, height, hfov_deg):
        """Square-pixel, centred pinhole from a horizontal field of view (Habitat's rule)."""
        f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, int(width), int(height))

    @property
    def hfov_deg(self):
        return float(np.degrees(2.0 * np.arctan((self.width / 2.0) / self.fx)))

    @property
    def matrix(self):
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


# --- rigid transforms ------------------------------------------------------------------

def quat_to_rot(qx, qy, qz, qw):
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("zero quaternion")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_matrix(t, q):
    """(tx,ty,tz), (qx,qy,qz,qw) -> 4x4 T mapping local -> world."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(*q)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def invert_pose(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def interpolate_pose(T0, T1, u):
    """Rigid pose a fraction `u` of the way from T0 to T1: lerp the translation, slerp the
    rotation. Works straight off the 4x4s (the source quaternions are not kept), by taking
    the relative rotation to axis-angle, scaling the angle, and composing it back on.
    """
    u = float(u)
    T = np.eye(4)
    T[:3, 3] = (1.0 - u) * T0[:3, 3] + u * T1[:3, 3]
    R_rel = T0[:3, :3].T @ T1[:3, :3]
    cos = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:  # no rotation between the samples
        T[:3, :3] = T0[:3, :3]
        return T
    axis = np.array([R_rel[2, 1] - R_rel[1, 2],
                     R_rel[0, 2] - R_rel[2, 0],
                     R_rel[1, 0] - R_rel[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:  # 180 deg: the off-diagonal trick degenerates, fall back to the nearer end
        T[:3, :3] = T0[:3, :3] if u < 0.5 else T1[:3, :3]
        return T
    axis, a = axis / n, angle * u
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_u = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)  # Rodrigues
    T[:3, :3] = T0[:3, :3] @ R_u
    return T


def transform_points(T, pts):
    """Apply 4x4 T to (N,3) points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


# --- pinhole ---------------------------------------------------------------------------

def project(points_cam, K: Intrinsics):
    """(N,3) camera-frame points -> (uv (N,2) pixels, z (N,)). uv is NaN where z <= Z_NEAR."""
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K.cx + p[:, 0] * K.fx / z
        v = K.cy + p[:, 1] * K.fy / z
    uv = np.stack([u, v], axis=1)
    uv[z <= Z_NEAR] = np.nan
    return uv, z


def backproject_depth(depth_m, K: Intrinsics, stride=4, max_depth=6.0):
    """Metric depth image (H,W) -> (N,3) camera-frame points; zeros/far pixels dropped."""
    d = np.asarray(depth_m, dtype=float)[::stride, ::stride]
    vs, us = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    ok = (d > 0) & (d <= max_depth)
    z = d[ok]
    x = (us[ok] - K.cx) * z / K.fx
    y = (vs[ok] - K.cy) * z / K.fy
    return np.stack([x, y, z], axis=1)


# --- floor plane -----------------------------------------------------------------------

@dataclass(frozen=True)
class FloorPlane:
    """n . p + d = 0 with n unit and oriented so the camera origin is above the floor (d > 0).
    For a level camera n ~ (0, -1, 0), so d is the camera height above the floor."""
    normal: tuple
    d: float
    inlier_frac: float

    @property
    def height(self):
        return float(self.d)

    @property
    def tilt_deg(self):
        """Angle between the floor normal and the camera's up axis (0,-1,0); 0 = level camera."""
        return float(np.degrees(np.arccos(np.clip(-self.normal[1], -1.0, 1.0))))


def fit_floor_plane(points_cam, n_iter=200, thresh=0.03, min_inlier_frac=0.15,
                    max_tilt_deg=35.0, seed=0):
    """RANSAC plane through (N,3) camera-frame points, restricted to floor-like planes.

    A candidate is floor-like when its normal is within `max_tilt_deg` of the camera's
    up axis and the camera sits above it. Returns the best FloorPlane, refined by a
    least-squares fit on the inliers, or None if no candidate reaches `min_inlier_frac`.
    """
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    if len(pts) < 50:
        return None
    rng = np.random.default_rng(seed)
    up = np.array([0.0, -1.0, 0.0])
    cos_max = np.cos(np.radians(max_tilt_deg))
    best, best_inl = None, None
    for _ in range(n_iter):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ a
        if d < 0:  # orient so the origin is on the positive side
            n, d = -n, -d
        if n @ up < cos_max:
            continue
        dist = np.abs(pts @ n + d)
        inl = dist < thresh
        if best is None or inl.sum() > best_inl.sum():
            best, best_inl = (n, d), inl
    if best is None or best_inl.mean() < min_inlier_frac:
        return None
    # least-squares refinement on inliers
    P = pts[best_inl]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    d = -n @ c
    if d < 0:
        n, d = -n, -d
    if n @ up < cos_max:
        n, d = best
    return FloorPlane(tuple(float(x) for x in n), float(d), float(best_inl.mean()))


def level_floor_plane(camera_height):
    """Flat-ground assumption for a level camera at `camera_height` metres (no depth)."""
    return FloorPlane((0.0, -1.0, 0.0), float(camera_height), 1.0)


def drop_to_plane(points_cam, plane: FloorPlane):
    """Orthogonal projection of (N,3) points onto the floor plane."""
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    n = np.asarray(plane.normal)
    s = p @ n + plane.d
    return p - s[:, None] * n[None, :]


# --- polyline helpers (Habitat generator rules) ----------------------------------------

def cumulative_arclength(positions):
    p = np.asarray(positions, dtype=float)
    p = p.reshape(len(p), -1)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def window_by_arclength(positions, start, length_m):
    """Index of the first sample at least `length_m` of path beyond `start`, or None
    if the trajectory ends first."""
    cum = cumulative_arclength(np.asarray(positions)[start:])
    hit = np.nonzero(cum >= length_m)[0]
    return int(start + hit[0]) if len(hit) else None


def densify(points, step_m=0.05):
    """Insert points so consecutive samples are at most `step_m` apart (endpoints kept)."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return p
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.array(out)


def front_runs(z, z_near=Z_NEAR):
    """Contiguous index runs with z > z_near, as (start, end_exclusive) tuples."""
    ok = np.asarray(z) > z_near
    runs, start = [], None
    for i, f in enumerate(ok):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(ok)))
    return runs


def occluded_mask(uv, z, depth_m, margin_m=OCCLUSION_MARGIN_M):
    """True where the depth image sees something closer than the point (minus margin).

    Out-of-frame points and depth holes (0) are NOT occluded — unknown counts as visible,
    matching how the Habitat labels treat the far floor.
    """
    uv = np.asarray(uv, dtype=float)
    z = np.asarray(z, dtype=float)
    H, W = depth_m.shape
    out = np.zeros(len(uv), dtype=bool)
    ok = np.isfinite(uv).all(axis=1)
    u = np.floor(uv[ok, 0]).astype(int)
    v = np.floor(uv[ok, 1]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.nonzero(ok)[0][inb]
    d = depth_m[v[inb], u[inb]]
    out[idx] = (d > 0) & (d < z[idx] - margin_m)
    return out
