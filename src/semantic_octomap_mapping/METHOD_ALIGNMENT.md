# S2P-SLAM method-to-code alignment

This package is the recovered implementation aligned with the submitted
manuscript.  The manuscript, rather than the legacy source file, is the
normative specification for the behavior summarized below.

## Estimator state and admitted factors

The iSAM2 state at keyframe `k` is `(T_WB, v_W, c, lambda)`.  The code stores
`ell = log(lambda)` as the scalar graph variable; additive updates of `ell`
implement the manuscript's multiplicative positive-state retraction exactly:
`lambda_new = lambda * exp(delta_ell)`.

The quadratic factor set contains LIO relative pose, pose--velocity
integration, hybrid kinematics, slip evolution, velocity random walk, period
evolution/measurement, ground plane, and the conditional five-dimensional
virtual rail.  The robust set contains structural periodicity, the
hemisphere-aligned `Unit3` VP factor, the closest-point aisle-centerline
factor, and visual flow.  A failed upstream gate creates no factor.

Two implementation details are intentionally immutable after factor creation:

- the aisle-local integer support association in
  `StructuralPeriodicityFactor`; and
- the sign selected by `HemisphereAlignedVpFactor`.

## Integrity switching

The LIO monitor tests stagnation, travel-polarity disagreement, and excessive
bounded-interval increments.  Two seconds of consecutive failure enters
fallback.  Fallback inserts no LIO relative-pose factor and uses encoder
longitudinal/yaw measurements, plus visual lateral velocity only when that
measurement passes its gate.  The same flow measurement is therefore not
inserted twice.  Recovery requires one second of converged, finite LIO output
with positive-definite covariance.  A fixed `T_WL'` alignment is established
at recovery and no LIO factor crosses the restart boundary.

The optional `/s2p/lio_converged` (`std_msgs/Bool`) input represents the
registration/deskew convergence gate.  If a deployment does not publish it,
the node starts with the flag true; recovery still requires a positive-definite
covariance in the LIO odometry message.

## Structural front end

- RGB-D depth is restricted to 0.30--8.0 m.
- World-height and HSV gates, gamma 0.7, CLAHE, optional VP floor exclusion,
  and row-dependent morphology produce the binary structure mask.
- LiDAR projection uses positive depth and mask membership only.  It never
  compares projected LiDAR range with RGB-D depth.
- Geometry-supported LiDAR clusters may pass without visual confirmation.
- Ground support uses body-frame RANSAC plane coefficients.
- Bilateral LiDAR slice midpoints and VP-derived pseudo-points at
  4.0:0.5:12.0 m produce a smoothed quadratic centerline.
- If the VP is unavailable, anisotropic bilateral wall-slice PCA supplies a
  fixed straight-centerline measurement; its common wall midpoint and heading
  are transformed into the active-aisle frame before factor creation.
- Sector-wise intensity thresholding, cluster geometry/PCA gates, a 0.05 m
  aisle histogram, autocorrelation, sub-bin interpolation, and 0.05 smoothing
  produce the local period measurement.
- Shi--Tomasi/LK flow is rotation compensated, converted to metric lateral
  velocity, aggregated with the retained depth-binned variance profile, and
  admitted only with at least 30 RANSAC inliers.
- In normal mode, image-interval rotation compensation uses a time-matched
  high-rate LIO orientation; a source change at fallback/recovery clears the
  tracker history so that rotations from different frames are never mixed.
- Cross-modal yaw correction is one-dimensional, safely gated, and used only
  by association/flow.  It is not a graph state or a mapping-pose correction.

## STSM separation and one-time integration

`SpatiotemporalSemanticMapper` receives only RGB-D packets for which a
posterior keyframe pose is available.  Each packet id is accepted at most
once.  An unconfirmed voxel stores its endpoint, ray, class label, timestamps,
hit/view counts, and cluster support, but performs **no** OctoMap hit or miss
update.  Promotion requires three hits, two views separated by at least
0.20 m, a 2.0 s observation span, and non-floor cluster support.  Approved
rays are replayed once in timestamp order.  Unpromoted records expire after
1.0 s without another observation.

Occupancy and semantic counts are separate.  The inverse sensor model uses
`p_hit=0.60`, `p_miss=0.40`, clamping probabilities 0.12/0.97, and occupancy
threshold 0.50.  Ground/structure/obstacle classification is evaluated only
for occupied voxels.  The published final map contains occupied voxels with at
least five occupied cells in their 26-neighborhood; rejected isolated voxels
remain in the internal evidence map.

## Input-frame assumption

The default `/cloud_registered_effect_world` input is assumed to contain the
deskewed cloud expressed in the current Faster-LIO local world frame.  The node
uses the time-matched LIO pose to express it in the body frame before feature
extraction.  If the cloud is already expressed in `body_frame`, no conversion
is applied.  A differently framed cloud must be converted upstream or the
callback extended with the appropriate calibrated transform.

## Build and run

The package targets Ubuntu 20.04, ROS Noetic, C++17, GTSAM, PCL, OpenCV, and
OctoMap.

```bash
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
roslaunch semantic_octomap_mapping mapping.launch
```

Set `S2P_LOG_DIR` to an existing writable directory to enable trajectory
logging.  The launch file contains the evaluation defaults; camera intrinsics
and the nominal `T_BC^0` must be replaced when the sensor installation changes.
The principal Full-minus-one switches are exposed as launch arguments; the
reported semantic removal disables both `enable_semantic_gate` and
`enable_stsm`.
