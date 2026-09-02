# Real-image evaluation: asset survey (2026-09-02)

Goal: test whether the Habitat-trained path models overfit to the renderer by scoring
them on **real** forward-facing robot frames. Labels come from the robot's own future
trajectory: for a frame at time *t*, the future positions (until the path is L m long)
are dropped to the floor plane, projected into frame *t*, occlusion-tested against depth
when available, and formatted exactly like the Habitat labels
(`{"path":[[x,y,v],...],"goal":[x,y,v]}`). Pipeline: `scripts/prepare_real_eval.py`,
`src/rover_vlm/real_data.py`, `src/rover_vlm/projection.py`.

Per frame the method needs: a 6-DoF camera pose (or base pose + camera extrinsic),
pinhole intrinsics, and the floor plane in camera coordinates. Depth additionally gives
Habitat-style visibility flags. Habitat reference camera: 0.8 m height, 90° HFOV,
512×512, level, paths 3–12 m.

## Chosen

| Set | Dataset | Platform | Poses | Floor plane | Visibility | Notes |
|---|---|---|---|---|---|---|
| `tum_pioneer` | [TUM RGB-D fr2 `pioneer_{360,slam,slam2,slam3}`](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) | Pioneer wheeled robot, Kinect, 640×480 @30 Hz, 58° HFOV | mocap pose of the colour camera's optical centre (`groundtruth.txt`, optical-frame convention — verified: forward→+z, world-up→−y) | RANSAC on registered depth (fitted height 0.60 ± 0.01 m) | depth occlusion test | CC BY 4.0, direct tgz (~5.5 GB). Industrial hall; short loopy drives, so horizons are 2–6 m. |
| `gnd_campus` | [GND](https://people.cs.gmu.edu/~xiao/Research/GND/) ([Dataverse](https://dataverse.orc.gmu.edu/dataset.xhtml?persistentId=doi:10.13021/orc2020/JUIW5F), [code](https://github.com/jingGM/GND)) | Clearpath Jackal, ZED2 rectified 640×360 @15 Hz, **101° HFOV** | `/odometry/filtered` (EKF, 50 Hz) + `tf_static` base→camera | normal from the ZED IMU gravity; **height is not in the bag — assumed 0.45 m** (`--cam-height`) | none (no depth): all visible | CC0. Bags chunked ~3 GB (`<campus>_chunkNN.bag`); only chunk01 carries `tf_static`. Campus zips hold LiDAR-SLAM outputs, no calibration. |

## Considered, not used (yet)

Indoor wheeled robot:
- **OpenLORIS-Scene** — D435i at ~1 m on a service robot, office/home/corridor/cafe/market, per-frame poses, `tf_static`/`extrinsics.yaml`, aligned depth. Form-gated, rosbag extraction. **Next indoor set** (user decision: after TUM).
- SACSoN/HuRoN, GO Stanford (ViNT `traj_data.pkl`) — 2-D odometry only, fisheye/spherical, 0.35 m camera.
- MIT Stata Center (PR2), Ground-Challenge, i2Nav-Robot (450 GB, GPL), Segway DRIVE (host flaky).

Outdoor / planetary:
- **CODa** (UT Austin) — best-documented extrinsics to base, tiny/small download tiers. Fallback if a calibrated outdoor set is needed.
- **RECON** — the only dataset with a published image-space projection (ViNT `data_config.yaml`: height 0.95 m, x-offset 0.45 m, K + distortion); 50 GB monolithic.
- **BASEPROD** (real rover, Bardenas, full tf tree, 2024), MADMAX / S3LI (DLR handheld, RTK) — planetary-analog, later phase.
- KITTI odometry / nuScenes mini — trivial projection, car embodiment (1.65 m camera): smoke test only.
- Not usable: Katwijk (ESA link dead), CPET (static scans), Wild-Places (no camera), TartanAir/LuSNAR (synthetic), FrodoBots (GPS only), Matterport3D/2D-3D-S/R2R (panorama stations).

Handheld / head-mounted (ScanNet, ScanNet++, ARKitScenes, BS3D, ETH3D, Aria, HoloSet):
home-like imagery closest to HM3D, but camera motion ≠ traversable path. Optional
"viewpoint-shifted" split only.

## Prior art on the recipe
- WayFAST / WayFASTER (https://github.com/matval/WayFAST): future poses → image via known
  K and extrinsics, for traversability masks; data + calibration released.
- ViNT `vint_train/visualizing/action_utils.py::project_points`: flat ground, zero camera
  rotation, a u-flip that only works when `cx == W/2`, drops out-of-frame points. Not
  reused; `rover_vlm.projection` keeps every point and clips like the Habitat labels.
