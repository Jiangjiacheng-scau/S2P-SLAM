# S2P-SLAM

Code snapshot for the manuscript:

**S2P-SLAM: Exploiting structural periodicity and unit sphere manifold constraints for semantic visual-LiDAR SLAM in caged poultry houses**

This repository provides the core ROS source snapshot and configuration files used for poultry-house localization and semantic mapping experiments.

## Repository layout

```text
src/
  faster-lio/                  LiDAR-inertial odometry frontend and poultry-house configuration
  semantic_octomap_mapping/    Semantic OctoMap mapping node, launch file, and RViz config
docs/
  demonstration_materials.md   Demonstration-material index and release notes
```

The full experimental system also used standard ROS driver packages for LiDAR, IMU/AHRS, and Intel RealSense RGB-D cameras. These driver packages are not vendored in this lightweight release to keep the repository small and reviewable.

## Key configuration files

```text
src/faster-lio/config/cp_lio.yaml
src/faster-lio/launch/cp_lio.launch
src/semantic_octomap_mapping/launch/mapping.launch
```

## Environment

The system was developed with:

- Ubuntu 20.04
- ROS Noetic
- PCL
- OpenCV
- Eigen
- GTSAM
- Intel RealSense SDK / ROS wrapper

## Build

Create a catkin workspace and place this repository content inside it:

```bash
mkdir -p ~/s2p_slam_ws/src
cp -r src/* ~/s2p_slam_ws/src/
cd ~/s2p_slam_ws
catkin_make
source devel/setup.bash
```

Install or clone the required sensor driver packages according to your hardware setup before building the full online system.

## Demonstration materials

Large raw videos, ROS bags, and point-cloud maps are intentionally excluded from Git. Demonstration videos and large experimental artifacts should be attached as GitHub Release assets or linked from:

```text
docs/demonstration_materials.md
```

## Excluded large files

The following file types are intentionally not tracked:

- ROS bags: `*.bag`
- Point clouds/maps: `*.pcd`, `*.ply`
- Raw videos: `*.mp4`, `*.avi`, `*.mov`, `*.mkv`
- Build products: `build/`, `devel/`, `install/`

