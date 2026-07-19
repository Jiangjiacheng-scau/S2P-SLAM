// S2P-SLAM: structural-periodicity and unit-sphere polarity constraints.
//
// This file is the executable reference implementation accompanying the
// manuscript.  The implementation deliberately keeps localization and map
// maintenance in separate data paths: only admitted measurements create graph
// factors, and the persistent semantic map never feeds measurements back to the
// estimator.

#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include <std_msgs/ColorRGBA.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <boost/array.hpp>

#include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <boost/bind/bind.hpp>
#include <boost/function.hpp>
#include <boost/make_shared.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace s2p {

using gtsam::Key;
using gtsam::Matrix;
using gtsam::Matrix3;
using gtsam::Matrix6;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Symbol;
using gtsam::Unit3;
using gtsam::Vector;
using gtsam::Vector1;
using gtsam::Vector2;
using gtsam::Vector3;
using gtsam::Vector4;
using gtsam::Vector6;

constexpr double kEpsilon = 1e-9;
constexpr double kPi = 3.141592653589793238462643383279502884;

double clamp(const double value, const double lo, const double hi) {
  return std::max(lo, std::min(hi, value));
}

double wrapPi(double angle) {
  while (angle >= kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

double alignUndirected(const double angle, const double reference) {
  const double a = wrapPi(angle);
  const double b = wrapPi(angle + kPi);
  return std::abs(wrapPi(a - reference)) <= std::abs(wrapPi(b - reference))
             ? a
             : b;
}

bool finiteVector(const Vector& value) { return value.array().isFinite().all(); }

bool finitePose(const Pose3& pose) {
  return finiteVector(Pose3::Logmap(pose));
}

bool validPoseMessage(const geometry_msgs::Pose& msg) {
  const std::array<double, 7> values{
      msg.position.x, msg.position.y, msg.position.z, msg.orientation.x,
      msg.orientation.y, msg.orientation.z, msg.orientation.w};
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  const double quaternion_norm = std::sqrt(
      msg.orientation.x * msg.orientation.x +
      msg.orientation.y * msg.orientation.y +
      msg.orientation.z * msg.orientation.z +
      msg.orientation.w * msg.orientation.w);
  return quaternion_norm >= kEpsilon;
}

Pose3 poseFromMsg(const geometry_msgs::Pose& msg) {
  const Eigen::Quaterniond q(msg.orientation.w, msg.orientation.x,
                             msg.orientation.y, msg.orientation.z);
  if (!q.coeffs().array().isFinite().all() || q.norm() < kEpsilon) {
    return Pose3();
  }
  const Eigen::Quaterniond normalized = q.normalized();
  return Pose3(Rot3(normalized.toRotationMatrix()),
               Point3(msg.position.x, msg.position.y, msg.position.z));
}

geometry_msgs::Pose poseToMsg(const Pose3& pose) {
  geometry_msgs::Pose msg;
  const auto q = pose.rotation().toQuaternion();
  msg.position.x = pose.x();
  msg.position.y = pose.y();
  msg.position.z = pose.z();
  msg.orientation.w = q.w();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
  return msg;
}

struct Config {
  // Topics and frames.
  std::string lio_odom_topic = "/Odometry";
  std::string lio_status_topic = "/s2p/lio_converged";
  std::string wheel_odom_topic = "/odom";
  std::string command_topic = "/cmd_vel";
  std::string lidar_topic = "/cloud_registered_effect_world";
  std::string rgb_topic = "/front_camera/color/image_raw";
  std::string depth_topic = "/front_camera/aligned_depth_to_color/image_raw";
  std::string world_frame = "s2p_world";
  std::string body_frame = "body";

  // Camera and nominal camera-to-body transform T_BC^0.
  double fx = 904.3275;
  double fy = 903.9418;
  double cx = 653.1630;
  double cy = 369.6624;
  Vector3 t_bc = (Vector3() << 0.10, 0.0, 0.0).finished();
  Vector3 rpy_bc = (Vector3() << -0.5 * kPi, 0.0, -0.5 * kPi).finished();

  // Feature extraction.
  double depth_min = 0.30;
  double depth_max = 8.0;
  double front_floor_height = 0.10;
  double front_ceiling_height = 3.50;
  double gamma = 0.70;
  int hsv_s_min = 40;
  int hsv_v_min = 30;
  int morph_far_w = 1;
  int morph_far_h = 3;
  int morph_near_w = 2;
  int morph_near_h = 10;
  int ground_ransac_iterations = 400;
  int min_image_line_length = 40;
  double structural_span = 0.03;
  double structural_span_dense = 0.02;
  double pca_linearity = 0.40;
  double histogram_resolution = 0.05;
  int radial_sectors = 12;
  double period_min = 1.50;
  double period_max = 1.80;
  double period_smoothing = 0.05;
  double period_confidence_min = 0.40;
  int period_support_min = 3;
  double centerline_smoothing = 0.10;
  double flow_ransac_threshold = 0.10;
  int flow_min_inliers = 30;
  double flow_confidence_smoothing = 0.50;
  double flow_sigma0 = 0.05;
  double yaw_calib_gain = 0.005;
  double yaw_calib_safe = 0.20;

  // Backend.
  double initial_period = 1.66;
  double initial_body_height = 0.50;
  double keyframe_translation = 0.05;
  double keyframe_rotation = 0.035;
  double keyframe_max_dt = 0.50;
  double slip_variance_v = 1.0e-4;
  double slip_variance_w = 1.5e-4;
  double slip_turn_gain = 0.01;
  double reversal_variance_multiplier = 9.0;
  double rail_covariance_threshold = 0.05;
  Vector3 rail_attitude_sigma = (Vector3() << 0.03, 0.03, 0.02).finished();
  double rail_lateral_sigma = 0.03;
  double rail_height_sigma = 0.05;
  double vp_sigma = 0.02;
  double lio_active_speed = 0.20;
  double lio_stagnation_speed = 0.05;
  double lio_polarity_speed = 0.05;
  double lio_max_translation = 0.30;
  double lio_max_rotation = 0.10;
  double lio_failure_time = 2.0;
  double lio_recovery_time = 1.0;
  double periodicity_sigma0 = 0.05;
  double periodicity_error_gain = 2.0;
  double periodicity_confidence_gain = 0.30;
  double period_lock_hold = 1.0;
  double aisle_heading_agreement = 0.20;

  // STSM.
  double voxel_resolution = 0.05;
  double map_ground_height = 0.10;
  double viewpoint_baseline = 0.20;
  int persistence_hits = 3;
  int persistence_views = 2;
  double persistence_time = 2.0;
  double candidate_decay = 1.0;
  double semantic_threshold = 0.75;
  double occupancy_threshold = 0.50;
  double occupancy_hit = 0.60;
  double occupancy_miss = 0.40;
  double occupancy_min = 0.12;
  double occupancy_max = 0.97;
  int occupied_neighbor_min = 5;
  int nonfloor_cluster_min = 3;
  int map_pixel_stride = 3;

  bool enable_periodicity = true;
  bool enable_rail = true;
  bool enable_vp = true;
  bool enable_line = true;
  bool enable_ground = true;
  bool enable_flow = true;
  bool enable_hybrid = true;
  bool enable_semantic_gate = true;
  bool enable_stsm = true;

  void load(ros::NodeHandle& nh) {
    nh.param("lio_odom_topic", lio_odom_topic, lio_odom_topic);
    nh.param("lio_status_topic", lio_status_topic, lio_status_topic);
    nh.param("wheel_odom_topic", wheel_odom_topic, wheel_odom_topic);
    nh.param("command_topic", command_topic, command_topic);
    nh.param("lidar_topic", lidar_topic, lidar_topic);
    nh.param("rgb_topic", rgb_topic, rgb_topic);
    nh.param("depth_topic", depth_topic, depth_topic);
    nh.param("global_frame_id", world_frame, world_frame);
    nh.param("body_frame_id", body_frame, body_frame);
    nh.param("cam_fx", fx, fx);
    nh.param("cam_fy", fy, fy);
    nh.param("cam_cx", cx, cx);
    nh.param("cam_cy", cy, cy);
    nh.param("t_bc_x", t_bc.x(), t_bc.x());
    nh.param("t_bc_y", t_bc.y(), t_bc.y());
    nh.param("t_bc_z", t_bc.z(), t_bc.z());
    nh.param("r_bc_roll", rpy_bc.x(), rpy_bc.x());
    nh.param("r_bc_pitch", rpy_bc.y(), rpy_bc.y());
    nh.param("r_bc_yaw", rpy_bc.z(), rpy_bc.z());
    nh.param("depth_min", depth_min, depth_min);
    nh.param("depth_max", depth_max, depth_max);
    nh.param("h_floor", front_floor_height, front_floor_height);
    nh.param("h_ceiling", front_ceiling_height, front_ceiling_height);
    nh.param("gamma", gamma, gamma);
    nh.param("hsv_s_min", hsv_s_min, hsv_s_min);
    nh.param("hsv_v_min", hsv_v_min, hsv_v_min);
    nh.param("initial_period", initial_period, initial_period);
    nh.param("initial_body_height", initial_body_height, initial_body_height);
    nh.param("period_min", period_min, period_min);
    nh.param("period_max", period_max, period_max);
    nh.param("period_confidence_min", period_confidence_min,
             period_confidence_min);
    nh.param("period_support_min", period_support_min, period_support_min);
    nh.param("period_smoothing", period_smoothing, period_smoothing);
    nh.param("flow_min_inliers", flow_min_inliers, flow_min_inliers);
    nh.param("flow_ransac_threshold", flow_ransac_threshold,
             flow_ransac_threshold);
    nh.param("slip_variance_v", slip_variance_v, slip_variance_v);
    nh.param("slip_variance_w", slip_variance_w, slip_variance_w);
    nh.param("slip_turn_gain", slip_turn_gain, slip_turn_gain);
    nh.param("rail_covariance_threshold", rail_covariance_threshold,
             rail_covariance_threshold);
    nh.param("lio_failure_time", lio_failure_time, lio_failure_time);
    nh.param("lio_recovery_time", lio_recovery_time, lio_recovery_time);
    nh.param("enable_periodicity_factor", enable_periodicity, enable_periodicity);
    nh.param("enable_virtual_rail_factor", enable_rail, enable_rail);
    nh.param("enable_vp_factor", enable_vp, enable_vp);
    nh.param("enable_line_factor", enable_line, enable_line);
    nh.param("enable_ground_factor", enable_ground, enable_ground);
    nh.param("enable_visual_flow", enable_flow, enable_flow);
    nh.param("enable_hybrid_factor", enable_hybrid, enable_hybrid);
    nh.param("enable_semantic_gate", enable_semantic_gate,
             enable_semantic_gate);
    nh.param("enable_stsm", enable_stsm, enable_stsm);
    nh.param("voxel_resolution", voxel_resolution, voxel_resolution);
    nh.param("map_ground_height", map_ground_height, map_ground_height);
    nh.param("temporal_hit_thresh", persistence_hits, persistence_hits);
    nh.param("temporal_trust_thresh", persistence_views, persistence_views);
    nh.param("temporal_static_life", persistence_time, persistence_time);
    nh.param("candidate_decay", candidate_decay, candidate_decay);
    nh.param("viewpoint_baseline", viewpoint_baseline, viewpoint_baseline);
    nh.param("semantic_thresh", semantic_threshold, semantic_threshold);
    nh.param("occupancy_thresh", occupancy_threshold, occupancy_threshold);
    nh.param("occupied_neighbor_min", occupied_neighbor_min, occupied_neighbor_min);
    nh.param("occupancy_hit", occupancy_hit, occupancy_hit);
    nh.param("occupancy_miss", occupancy_miss, occupancy_miss);
    nh.param("occupancy_min", occupancy_min, occupancy_min);
    nh.param("occupancy_max", occupancy_max, occupancy_max);
    nh.param("map_pixel_stride", map_pixel_stride, map_pixel_stride);
  }

  Pose3 nominalTbc() const {
    return Pose3(Rot3::Ypr(rpy_bc.z(), rpy_bc.y(), rpy_bc.x()), Point3(t_bc));
  }
};

struct WheelObservation {
  ros::Time stamp;
  double velocity = 0.0;
  double yaw_rate = 0.0;
  double commanded_velocity = 0.0;
  double commanded_yaw_rate = 0.0;
  bool valid = false;
};

struct VpObservation {
  ros::Time stamp;
  bool valid = false;
  cv::Point2f pixel{0.0F, 0.0F};
  Unit3 direction_camera = Unit3(Vector3(0.0, 0.0, 1.0));
  int inliers = 0;
  double confidence = 0.0;
  double viewing_angle = kPi;
};

struct FlowObservation {
  ros::Time stamp;
  bool valid = false;
  double lateral_velocity = 0.0;
  double confidence = 0.0;
  double variance = 1.0;
  int inliers = 0;
  int total = 0;
};

struct GroundObservation {
  ros::Time stamp;
  bool valid = false;
  Vector4 plane_body = Vector4(0.0, 0.0, 1.0, 0.0);
  int inliers = 0;
};

struct AisleFrame {
  bool valid = false;
  std::uint64_t id = 0;
  Rot3 r_wa;
  Point3 origin_w{0.0, 0.0, 0.0};
  double phase = 0.0;
  bool phase_initialized = false;
  double nominal_height = 0.0;

  Point3 toAisle(const Point3& point_w) const {
    return r_wa.unrotate(point_w - origin_w);
  }

  Vector3 axisWorld() const { return r_wa.rotate(Vector3(1.0, 0.0, 0.0)); }
};

struct CenterlineObservation {
  ros::Time stamp;
  bool valid = false;
  bool wall_fallback = false;
  std::uint64_t aisle_id = 0;
  AisleFrame frame;
  Vector3 coefficients = Vector3::Zero();  // [a,b,c] in y=a*x^2+b*x+c.
  int support = 0;
  double lateral_sigma = 0.08;
  double heading_sigma = 0.05;
};

struct PeriodObservation {
  ros::Time stamp;
  bool valid = false;
  bool event = false;
  bool semantic_gate = false;
  bool phase_gate = false;
  bool coherent_measurement = false;
  double spacing = 1.66;
  double confidence = 0.0;
  int support = 0;
};

struct FeatureBundle {
  VpObservation vp;
  FlowObservation flow;
  GroundObservation ground;
  CenterlineObservation line;
  PeriodObservation period;
  Rot3 r_cb_corrected;
  bool lidar_valid = false;
};

struct PosteriorPose {
  ros::Time stamp;
  Pose3 pose;
  Vector3 velocity = Vector3::Zero();
  Matrix6 covariance = Matrix6::Identity();
  std::size_t keyframe = 0;
};

// ---------------------------------------------------------------------------
// Factor definitions.  Discrete associations and measurement covariances are
// constructor arguments so that iSAM2 relinearization cannot change them.
// ---------------------------------------------------------------------------

class PoseVelocityIntegrationFactor
    : public gtsam::NoiseModelFactor4<Pose3, Pose3, Vector3, Vector3> {
 public:
  PoseVelocityIntegrationFactor(Key xi, Key xj, Key vi, Key vj, double dt,
                                const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor4(model, xi, xj, vi, vj), dt_(dt) {}

  Vector error(const Pose3& xi, const Pose3& xj, const Vector3& vi,
               const Vector3& vj) const {
    return xj.translation() - xi.translation() - 0.5 * dt_ * (vi + vj);
  }

  Vector evaluateError(const Pose3& xi, const Pose3& xj, const Vector3& vi,
                       const Vector3& vj, boost::optional<Matrix&> h1 = boost::none,
                       boost::optional<Matrix&> h2 = boost::none,
                       boost::optional<Matrix&> h3 = boost::none,
                       boost::optional<Matrix&> h4 = boost::none) const override {
    boost::function<Vector(const Pose3&, const Pose3&, const Vector3&, const Vector3&)> f =
        [this](const Pose3& a, const Pose3& b, const Vector3& c, const Vector3& d) {
          return error(a, b, c, d);
        };
    if (h1) *h1 = gtsam::numericalDerivative41<Vector, Pose3, Pose3, Vector3, Vector3>(f, xi, xj, vi, vj);
    if (h2) *h2 = gtsam::numericalDerivative42<Vector, Pose3, Pose3, Vector3, Vector3>(f, xi, xj, vi, vj);
    if (h3) *h3 = gtsam::numericalDerivative43<Vector, Pose3, Pose3, Vector3, Vector3>(f, xi, xj, vi, vj);
    if (h4) *h4 = gtsam::numericalDerivative44<Vector, Pose3, Pose3, Vector3, Vector3>(f, xi, xj, vi, vj);
    return error(xi, xj, vi, vj);
  }

 private:
  double dt_;
};

class HybridKinematicFactor
    : public gtsam::NoiseModelFactor4<Pose3, Pose3, Vector3, Vector2> {
 public:
  HybridKinematicFactor(Key xi, Key xj, Key vj, Key cj, double dt,
                        double encoder_velocity, double lateral_velocity,
                        bool use_lateral, double encoder_yaw_rate,
                        const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor4(model, xi, xj, vj, cj),
        dt_(dt),
        encoder_velocity_(encoder_velocity),
        lateral_velocity_(lateral_velocity),
        use_lateral_(use_lateral),
        encoder_yaw_rate_(encoder_yaw_rate) {}

  Vector error(const Pose3& xi, const Pose3& xj, const Vector3& velocity_w,
               const Vector2& scales) const {
    const Vector3 velocity_b = xj.rotation().unrotate(velocity_w);
    const Vector3 rotation_vector = Rot3::Logmap(xi.rotation().between(xj.rotation()));
    const double yaw_rate = rotation_vector.z() / std::max(dt_, kEpsilon);
    Vector residual(use_lateral_ ? 4 : 3);
    residual(0) = velocity_b.x() - scales.x() * encoder_velocity_;
    if (use_lateral_) {
      residual(1) = velocity_b.y() - lateral_velocity_;
      residual(2) = velocity_b.z();
      residual(3) = yaw_rate - scales.y() * encoder_yaw_rate_;
    } else {
      residual(1) = velocity_b.z();
      residual(2) = yaw_rate - scales.y() * encoder_yaw_rate_;
    }
    return residual;
  }

  Vector evaluateError(const Pose3& xi, const Pose3& xj, const Vector3& vj,
                       const Vector2& cj, boost::optional<Matrix&> h1 = boost::none,
                       boost::optional<Matrix&> h2 = boost::none,
                       boost::optional<Matrix&> h3 = boost::none,
                       boost::optional<Matrix&> h4 = boost::none) const override {
    boost::function<Vector(const Pose3&, const Pose3&, const Vector3&, const Vector2&)> f =
        [this](const Pose3& a, const Pose3& b, const Vector3& c, const Vector2& d) {
          return error(a, b, c, d);
        };
    if (h1) *h1 = gtsam::numericalDerivative41<Vector, Pose3, Pose3, Vector3, Vector2>(f, xi, xj, vj, cj);
    if (h2) *h2 = gtsam::numericalDerivative42<Vector, Pose3, Pose3, Vector3, Vector2>(f, xi, xj, vj, cj);
    if (h3) *h3 = gtsam::numericalDerivative43<Vector, Pose3, Pose3, Vector3, Vector2>(f, xi, xj, vj, cj);
    if (h4) *h4 = gtsam::numericalDerivative44<Vector, Pose3, Pose3, Vector3, Vector2>(f, xi, xj, vj, cj);
    return error(xi, xj, vj, cj);
  }

 private:
  double dt_;
  double encoder_velocity_;
  double lateral_velocity_;
  bool use_lateral_;
  double encoder_yaw_rate_;
};

class PeriodEvolutionFactor : public gtsam::NoiseModelFactor2<double, double> {
 public:
  PeriodEvolutionFactor(Key li, Key lj, const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, li, lj) {}

  Vector error(const double& log_i, const double& log_j) const {
    return Vector1(std::exp(log_j) - std::exp(log_i));
  }

  Vector evaluateError(const double& log_i, const double& log_j,
                       boost::optional<Matrix&> h1 = boost::none,
                       boost::optional<Matrix&> h2 = boost::none) const override {
    if (h1) *h1 = (Matrix(1, 1) << -std::exp(log_i)).finished();
    if (h2) *h2 = (Matrix(1, 1) << std::exp(log_j)).finished();
    return error(log_i, log_j);
  }
};

class PeriodMeasurementFactor : public gtsam::NoiseModelFactor1<double> {
 public:
  PeriodMeasurementFactor(Key key, double measured,
                          const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor1(model, key), measured_(measured) {}

  Vector evaluateError(const double& log_period,
                       boost::optional<Matrix&> h = boost::none) const override {
    if (h) *h = (Matrix(1, 1) << std::exp(log_period)).finished();
    return Vector1(std::exp(log_period) - measured_);
  }

 private:
  double measured_;
};

class StructuralPeriodicityFactor
    : public gtsam::NoiseModelFactor2<Pose3, double> {
 public:
  StructuralPeriodicityFactor(Key pose_key, Key period_key, AisleFrame frame,
                              std::int64_t fixed_index,
                              const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, pose_key, period_key),
        frame_(std::move(frame)),
        fixed_index_(fixed_index) {}

  Vector error(const Pose3& pose, const double& log_period) const {
    const double x_long = frame_.toAisle(pose.translation()).x();
    return Vector1(x_long - frame_.phase -
                   static_cast<double>(fixed_index_) * std::exp(log_period));
  }

  Vector evaluateError(const Pose3& pose, const double& log_period,
                       boost::optional<Matrix&> h1 = boost::none,
                       boost::optional<Matrix&> h2 = boost::none) const override {
    boost::function<Vector(const Pose3&, const double&)> f =
        [this](const Pose3& p, const double& l) { return error(p, l); };
    if (h1) *h1 = gtsam::numericalDerivative21<Vector, Pose3, double>(f, pose, log_period);
    if (h2) *h2 = gtsam::numericalDerivative22<Vector, Pose3, double>(f, pose, log_period);
    return error(pose, log_period);
  }

 private:
  AisleFrame frame_;
  std::int64_t fixed_index_;
};

class GroundPlaneFactor : public gtsam::NoiseModelFactor1<Pose3> {
 public:
  GroundPlaneFactor(Key key, Vector4 measured_plane_body,
                    const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor1(model, key), measured_(std::move(measured_plane_body)) {}

  Vector error(const Pose3& pose) const {
    const Vector4 world_plane(0.0, 0.0, 1.0, 0.0);
    return pose.matrix().transpose() * world_plane - measured_;
  }

  Vector evaluateError(const Pose3& pose,
                       boost::optional<Matrix&> h = boost::none) const override {
    boost::function<Vector(const Pose3&)> f =
        [this](const Pose3& p) { return error(p); };
    if (h) *h = gtsam::numericalDerivative11<Vector, Pose3>(f, pose);
    return error(pose);
  }

 private:
  Vector4 measured_;
};

class AisleCenterlineFactor : public gtsam::NoiseModelFactor1<Pose3> {
 public:
  AisleCenterlineFactor(Key key, CenterlineObservation measurement,
                       const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor1(model, key), measurement_(std::move(measurement)) {}

  Vector error(const Pose3& pose) const {
    const Point3 p_a = measurement_.frame.toAisle(pose.translation());
    const double a = measurement_.coefficients.x();
    const double b = measurement_.coefficients.y();
    const double c = measurement_.coefficients.z();
    double x_star = p_a.x();
    for (int iteration = 0; iteration < 12; ++iteration) {
      const double f = a * x_star * x_star + b * x_star + c;
      const double fp = 2.0 * a * x_star + b;
      const double gradient = (x_star - p_a.x()) + (f - p_a.y()) * fp;
      const double hessian = 1.0 + fp * fp + 2.0 * a * (f - p_a.y());
      if (std::abs(hessian) < 1e-8) break;
      const double step = clamp(gradient / hessian, -0.5, 0.5);
      x_star -= step;
      if (std::abs(step) < 1e-7) break;
    }
    const double y_star = a * x_star * x_star + b * x_star + c;
    const double slope = 2.0 * a * x_star + b;
    const Vector2 normal = Vector2(-slope, 1.0).normalized();
    const double lateral = normal.dot(Vector2(p_a.x() - x_star, p_a.y() - y_star));
    const Rot3 r_ab = measurement_.frame.r_wa.between(pose.rotation());
    const double heading = wrapPi(r_ab.yaw() - std::atan(slope));
    return (Vector2() << lateral, heading).finished();
  }

  Vector evaluateError(const Pose3& pose,
                       boost::optional<Matrix&> h = boost::none) const override {
    boost::function<Vector(const Pose3&)> f =
        [this](const Pose3& p) { return error(p); };
    if (h) *h = gtsam::numericalDerivative11<Vector, Pose3>(f, pose);
    return error(pose);
  }

 private:
  CenterlineObservation measurement_;
};

class HemisphereAlignedVpFactor : public gtsam::NoiseModelFactor1<Pose3> {
 public:
  HemisphereAlignedVpFactor(Key key, Unit3 measurement, Rot3 r_cb,
                            Vector3 aisle_axis_world, int fixed_sign,
                            const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor1(model, key),
        measurement_(std::move(measurement)),
        r_cb_(std::move(r_cb)),
        aisle_axis_world_(std::move(aisle_axis_world)),
        fixed_sign_(fixed_sign >= 0 ? 1 : -1) {}

  Vector error(const Pose3& pose) const {
    Vector3 predicted = r_cb_.rotate(pose.rotation().unrotate(aisle_axis_world_));
    predicted *= static_cast<double>(fixed_sign_);
    if (predicted.norm() < kEpsilon || !predicted.array().isFinite().all()) {
      return Vector2::Constant(1e3);
    }
    const Unit3 predicted_unit(predicted);
    const Vector2 residual = measurement_.errorVector(predicted_unit);
    return residual.array().isFinite().all() ? residual : Vector2::Constant(1e3);
  }

  Vector evaluateError(const Pose3& pose,
                       boost::optional<Matrix&> h = boost::none) const override {
    boost::function<Vector(const Pose3&)> f =
        [this](const Pose3& p) { return error(p); };
    if (h) *h = gtsam::numericalDerivative11<Vector, Pose3>(f, pose);
    return error(pose);
  }

 private:
  Unit3 measurement_;
  Rot3 r_cb_;
  Vector3 aisle_axis_world_;
  int fixed_sign_;
};

class VisualFlowFactor : public gtsam::NoiseModelFactor2<Pose3, Vector3> {
 public:
  VisualFlowFactor(Key pose_key, Key velocity_key, double measured_lateral,
                   const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor2(model, pose_key, velocity_key),
        measured_lateral_(measured_lateral) {}

  Vector error(const Pose3& pose, const Vector3& velocity_w) const {
    return Vector1(pose.rotation().unrotate(velocity_w).y() - measured_lateral_);
  }

  Vector evaluateError(const Pose3& pose, const Vector3& velocity_w,
                       boost::optional<Matrix&> h1 = boost::none,
                       boost::optional<Matrix&> h2 = boost::none) const override {
    boost::function<Vector(const Pose3&, const Vector3&)> f =
        [this](const Pose3& p, const Vector3& v) { return error(p, v); };
    if (h1) *h1 = gtsam::numericalDerivative21<Vector, Pose3, Vector3>(f, pose, velocity_w);
    if (h2) *h2 = gtsam::numericalDerivative22<Vector, Pose3, Vector3>(f, pose, velocity_w);
    return error(pose, velocity_w);
  }

 private:
  double measured_lateral_;
};

class VirtualRailFactor : public gtsam::NoiseModelFactor1<Pose3> {
 public:
  VirtualRailFactor(Key key, AisleFrame frame,
                    const gtsam::SharedNoiseModel& model)
      : NoiseModelFactor1(model, key), frame_(std::move(frame)) {}

  Vector error(const Pose3& pose) const {
    const Vector3 attitude = Rot3::Logmap(frame_.r_wa.between(pose.rotation()));
    const Point3 p_a = frame_.toAisle(pose.translation());
    Vector residual(5);
    residual << attitude, p_a.y(), p_a.z() - frame_.nominal_height;
    return residual;
  }

  Vector evaluateError(const Pose3& pose,
                       boost::optional<Matrix&> h = boost::none) const override {
    boost::function<Vector(const Pose3&)> f =
        [this](const Pose3& p) { return error(p); };
    if (h) *h = gtsam::numericalDerivative11<Vector, Pose3>(f, pose);
    return error(pose);
  }

 private:
  AisleFrame frame_;
};

// ---------------------------------------------------------------------------
// Incremental backend.
// ---------------------------------------------------------------------------

class FactorGraphBackend {
 public:
  enum class Mode { kLio, kFallback };

  explicit FactorGraphBackend(const Config& config) : config_(config) {
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.05;
    parameters.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(parameters);
  }

  std::optional<PosteriorPose> process(const nav_msgs::Odometry& lio_msg,
                                       const WheelObservation& wheel,
                                       const FeatureBundle& features,
                                       bool lio_status_converged) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Pose3 lio_local = poseFromMsg(lio_msg.pose.pose);
    if (!finitePose(lio_local)) {
      ROS_WARN_THROTTLE(1.0, "S2P backend rejected a non-finite LIO pose");
      return std::nullopt;
    }

    if (!initialized_) return initialize(lio_msg.header.stamp, lio_local);

    const double dt_lio = (lio_msg.header.stamp - last_lio_stamp_).toSec();
    if (dt_lio <= 0.0) return std::nullopt;
    const double since_keyframe =
        (lio_msg.header.stamp - last_keyframe_stamp_).toSec();
    if (since_keyframe <= 0.0) return std::nullopt;
    // Integrity tests use the same bounded keyframe interval that would feed
    // the relative-pose factor, while dt_lio below advances persistence timers
    // without double-counting the growing interval.
    const Pose3 delta_lio = last_keyframe_lio_local_.between(lio_local);
    const double delta_translation = delta_lio.translation().norm();
    const double delta_rotation = Rot3::Logmap(delta_lio.rotation()).norm();
    const double lio_forward_velocity = delta_lio.x() / since_keyframe;

    const bool stagnation = std::abs(wheel.velocity) > config_.lio_active_speed &&
                            std::abs(lio_forward_velocity) < config_.lio_stagnation_speed;
    const bool polarity = std::abs(wheel.velocity) > config_.lio_active_speed &&
                          std::abs(lio_forward_velocity) > config_.lio_polarity_speed &&
                          wheel.velocity * lio_forward_velocity < 0.0;
    const bool excessive = delta_translation > config_.lio_max_translation ||
                           delta_rotation > config_.lio_max_rotation;
    const bool lio_failure = stagnation || polarity || excessive;
    failure_duration_ = lio_failure ? failure_duration_ + dt_lio : 0.0;

    if (mode_ == Mode::kLio && failure_duration_ >= config_.lio_failure_time) {
      mode_ = Mode::kFallback;
      recovery_duration_ = 0.0;
      ROS_WARN("S2P backend entered encoder/flow fallback after %.2f s of LIO integrity failures",
               failure_duration_);
    }

    const bool convergence_gate = lio_status_converged && !lio_failure &&
                                  covariancePositiveDefinite(lio_msg.pose.covariance);
    bool recover_after_update = false;
    if (mode_ == Mode::kFallback) {
      recovery_duration_ = convergence_gate ? recovery_duration_ + dt_lio : 0.0;
      recover_after_update = recovery_duration_ >= config_.lio_recovery_time;
    }

    // Keyframe motion is measured from the last admitted keyframe, not from
    // the immediately preceding high-rate LIO callback.
    const Pose3 previous_aligned_lio =
        world_from_lio_.compose(last_keyframe_lio_local_);
    const Pose3 current_aligned_lio = world_from_lio_.compose(lio_local);
    const Pose3 keyframe_lio_delta =
        previous_aligned_lio.between(current_aligned_lio);
    const double predicted_encoder_distance = std::abs(wheel.velocity) * since_keyframe;
    const bool make_keyframe = since_keyframe >= config_.keyframe_max_dt ||
                               (mode_ == Mode::kLio
                                    ? keyframe_lio_delta.translation().norm() >= config_.keyframe_translation ||
                                          Rot3::Logmap(keyframe_lio_delta.rotation()).norm() >= config_.keyframe_rotation
                                    : predicted_encoder_distance >= config_.keyframe_translation ||
                                          std::abs(wheel.yaw_rate) * since_keyframe >= config_.keyframe_rotation) ||
                               features.period.event;

    last_lio_stamp_ = lio_msg.header.stamp;
    if (!make_keyframe) return std::nullopt;

    const double dt = (lio_msg.header.stamp - last_keyframe_stamp_).toSec();
    if (dt <= 0.0) return std::nullopt;
    const std::size_t previous = keyframe_;
    ++keyframe_;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    const bool use_lio_interval = mode_ == Mode::kLio;
    const Pose3 lio_interval = previous_aligned_lio.between(current_aligned_lio);

    Pose3 predicted_pose;
    double lio_lateral_velocity = 0.0;
    if (use_lio_interval) {
      predicted_pose = current_pose_.compose(lio_interval);
      lio_lateral_velocity = lio_interval.y() / dt;
      const Vector6 lio_sigmas = (Vector6() << 0.02, 0.02, 0.03, 0.03, 0.03, 0.03).finished();
      graph.add(gtsam::BetweenFactor<Pose3>(X(previous), X(keyframe_), lio_interval,
                                            gtsam::noiseModel::Diagonal::Sigmas(lio_sigmas)));
    } else {
      const Pose3 encoder_delta(Rot3::Ypr(wheel.yaw_rate * dt, 0.0, 0.0),
                                Point3(wheel.velocity * dt, 0.0, 0.0));
      predicted_pose = current_pose_.compose(encoder_delta);
    }

    const bool flow_admitted = config_.enable_flow && features.flow.valid &&
                               features.flow.inliers >= config_.flow_min_inliers &&
                               features.flow.confidence > 0.0 &&
                               std::isfinite(features.flow.lateral_velocity);
    const bool use_lateral = use_lio_interval || flow_admitted;
    const double lateral_measurement = use_lio_interval
                                           ? lio_lateral_velocity
                                           : features.flow.lateral_velocity;
    const Vector3 body_velocity(current_scales_.x() * wheel.velocity,
                                use_lateral ? lateral_measurement : 0.0, 0.0);
    const Vector3 predicted_velocity = predicted_pose.rotation().rotate(body_velocity);

    values.insert(X(keyframe_), predicted_pose);
    values.insert(V(keyframe_), predicted_velocity);
    values.insert(C(keyframe_), current_scales_);
    values.insert(L(keyframe_), current_log_period_);

    graph.add(boost::make_shared<PoseVelocityIntegrationFactor>(
        X(previous), X(keyframe_), V(previous), V(keyframe_), dt,
        gtsam::noiseModel::Isotropic::Sigma(3, 0.08)));

    const bool reversal = wheel.commanded_velocity * previous_encoder_velocity_ < 0.0;
    const double sigma_x = 0.15 * std::sqrt(reversal ? config_.reversal_variance_multiplier : 1.0);
    Vector hybrid_sigmas(use_lateral ? 4 : 3);
    if (use_lateral) {
      const double sigma_y = use_lio_interval
                                 ? 0.08
                                 : std::sqrt(std::max(features.flow.variance, 1e-8));
      hybrid_sigmas << sigma_x, sigma_y, 0.10, 0.08;
    } else {
      hybrid_sigmas << sigma_x, 0.10, 0.08;
    }
    if (config_.enable_hybrid && wheel.valid) {
      graph.add(boost::make_shared<HybridKinematicFactor>(
          X(previous), X(keyframe_), V(keyframe_), C(keyframe_), dt,
          wheel.velocity, lateral_measurement, use_lateral, wheel.yaw_rate,
          gtsam::noiseModel::Diagonal::Sigmas(hybrid_sigmas)));
    }

    const double slip_sigma_v = std::sqrt(config_.slip_variance_v +
                                           config_.slip_turn_gain *
                                               wheel.commanded_yaw_rate * wheel.commanded_yaw_rate);
    const double slip_sigma_w = std::sqrt(config_.slip_variance_w);
    graph.add(gtsam::BetweenFactor<Vector2>(
        C(previous), C(keyframe_), Vector2::Zero(),
        gtsam::noiseModel::Diagonal::Sigmas(Vector2(slip_sigma_v, slip_sigma_w))));
    graph.add(gtsam::BetweenFactor<Vector3>(
        V(previous), V(keyframe_), Vector3::Zero(),
        gtsam::noiseModel::Isotropic::Sigma(3, 0.35)));
    graph.add(boost::make_shared<PeriodEvolutionFactor>(
        L(previous), L(keyframe_), gtsam::noiseModel::Isotropic::Sigma(1, 0.02)));

    const bool lidar_structural_available = use_lio_interval && !lio_failure &&
                                            lio_status_converged &&
                                            features.lidar_valid;
    if (lidar_structural_available) {
      updateAisleFrame(predicted_pose, features, wheel, lio_msg.header.stamp);
    }
    const Matrix3 predicted_position_covariance = predictedPositionCovariance(dt);
    const double rail_statistic = railSubspaceStatistic(predicted_position_covariance);

    bool period_event_inserted = false;
    if (lidar_structural_available && config_.enable_periodicity && aisle_.valid &&
        features.period.valid &&
        features.period.confidence >= config_.period_confidence_min &&
        features.period.support >= config_.period_support_min &&
        features.period.semantic_gate) {
      if (features.period.coherent_measurement &&
          features.period.spacing >= config_.period_min &&
          features.period.spacing <= config_.period_max) {
        const double sigma = 0.03 / std::max(features.period.confidence, 0.1);
        graph.add(boost::make_shared<PeriodMeasurementFactor>(
            L(keyframe_), features.period.spacing,
            gtsam::noiseModel::Isotropic::Sigma(1, sigma)));
      }

      if (features.period.event && features.period.phase_gate) {
        const double x_long = aisle_.toAisle(predicted_pose.translation()).x();
        const double predicted_period = std::exp(current_log_period_);
        std::int64_t fixed_index = 0;
        if (!aisle_.phase_initialized) {
          aisle_.phase = std::fmod(x_long, predicted_period);
          if (aisle_.phase < 0.0) aisle_.phase += predicted_period;
          fixed_index = static_cast<std::int64_t>(std::llround(
              (x_long - aisle_.phase) / std::max(predicted_period, 1e-6)));
          aisle_.phase_initialized = true;
          last_period_index_ = fixed_index;
        } else {
          fixed_index = static_cast<std::int64_t>(std::llround(
              (x_long - aisle_.phase) / std::max(predicted_period, 1e-6)));
        }
        const double snap_error = std::abs(x_long - aisle_.phase -
                                           static_cast<double>(fixed_index) * predicted_period);
        const std::int64_t index_step = fixed_index - last_period_index_;
        const double signed_motion = x_long -
            aisle_.toAisle(current_pose_.translation()).x();
        const bool polarity_ok = index_step == 0 ||
                                 static_cast<double>(index_step) * signed_motion >= 0.0;
        const bool travel_ok = std::abs(index_step) <= std::max<std::int64_t>(
            1, static_cast<std::int64_t>(std::ceil(
                   std::abs(signed_motion) / predicted_period)) + 1);
        if (polarity_ok && travel_ok) {
          const double sigma = config_.periodicity_sigma0 +
                               config_.periodicity_error_gain * snap_error +
                               config_.periodicity_confidence_gain *
                                   (1.0 - features.period.confidence);
          auto model = gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(1.345),
              gtsam::noiseModel::Isotropic::Sigma(1, sigma));
          graph.add(boost::make_shared<StructuralPeriodicityFactor>(
              X(keyframe_), L(keyframe_), aisle_, fixed_index, model));
          last_period_index_ = fixed_index;
          last_period_lock_stamp_ = lio_msg.header.stamp;
          period_event_inserted = true;
        }
      }
    }

    if (lidar_structural_available && config_.enable_ground &&
        features.ground.valid) {
      graph.add(boost::make_shared<GroundPlaneFactor>(
          X(keyframe_), features.ground.plane_body,
          gtsam::noiseModel::Diagonal::Sigmas(
              (Vector4() << 0.05, 0.05, 0.05, 0.05).finished())));
    }

    if (lidar_structural_available && config_.enable_line &&
        features.line.valid && aisle_.valid &&
        features.line.aisle_id == aisle_.id) {
      auto model = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(1.345),
          gtsam::noiseModel::Diagonal::Sigmas(
              Vector2(features.line.lateral_sigma, features.line.heading_sigma)));
      graph.add(boost::make_shared<AisleCenterlineFactor>(
          X(keyframe_), features.line, model));
    }

    if (config_.enable_vp && features.vp.valid && aisle_.valid &&
        features.vp.viewing_angle <= 78.0 * kPi / 180.0) {
      const Vector3 predicted_camera = features.r_cb_corrected.rotate(
          predicted_pose.rotation().unrotate(aisle_.axisWorld()));
      const int fixed_sign = predicted_camera.dot(features.vp.direction_camera.unitVector()) >= 0.0
                                 ? 1
                                 : -1;
      const double sigma = config_.vp_sigma / std::max(features.vp.confidence, 0.05);
      auto model = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(1.345),
          gtsam::noiseModel::Isotropic::Sigma(2, sigma));
      graph.add(boost::make_shared<HemisphereAlignedVpFactor>(
          X(keyframe_), features.vp.direction_camera, features.r_cb_corrected,
          aisle_.axisWorld(), fixed_sign, model));
    }

    if (flow_admitted && use_lio_interval) {
      auto model = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(1.345),
          gtsam::noiseModel::Isotropic::Sigma(
              1, std::sqrt(std::max(features.flow.variance, 1e-8))));
      graph.add(boost::make_shared<VisualFlowFactor>(
          X(keyframe_), V(keyframe_), features.flow.lateral_velocity, model));
    }

    const bool periodic_lock = !last_period_lock_stamp_.isZero() &&
                               (lio_msg.header.stamp - last_period_lock_stamp_).toSec() <=
                                   config_.period_lock_hold;
    const bool straight_aisle = lidar_structural_available &&
                                straightAisleGate(features, wheel,
                                                  predicted_pose);
    if (config_.enable_rail && aisle_.valid && straight_aisle && !periodic_lock &&
        rail_statistic > config_.rail_covariance_threshold) {
      Vector rail_sigmas(5);
      rail_sigmas << config_.rail_attitude_sigma, config_.rail_lateral_sigma,
          config_.rail_height_sigma;
      graph.add(boost::make_shared<VirtualRailFactor>(
          X(keyframe_), aisle_, gtsam::noiseModel::Diagonal::Sigmas(rail_sigmas)));
    }

    try {
      isam_->update(graph, values);
      isam_->update();
      current_pose_ = isam_->calculateEstimate<Pose3>(X(keyframe_));
      current_velocity_ = isam_->calculateEstimate<Vector3>(V(keyframe_));
      current_scales_ = isam_->calculateEstimate<Vector2>(C(keyframe_));
      current_log_period_ = isam_->calculateEstimate<double>(L(keyframe_));
      current_covariance_ = isam_->marginalCovariance(X(keyframe_));
    } catch (const std::exception& exception) {
      ROS_ERROR("S2P iSAM2 update failed at keyframe %zu: %s", keyframe_, exception.what());
      --keyframe_;
      return std::nullopt;
    }

    if (!finitePose(current_pose_) || !finiteVector(current_velocity_) ||
        !finiteVector(current_scales_) || !std::isfinite(current_log_period_)) {
      ROS_ERROR("S2P backend produced a non-finite posterior; update is not published");
      return std::nullopt;
    }

    if (recover_after_update) {
      world_from_lio_ = current_pose_.compose(lio_local.inverse());
      last_keyframe_lio_local_ = lio_local;
      mode_ = Mode::kLio;
      failure_duration_ = 0.0;
      recovery_duration_ = 0.0;
      ROS_INFO("S2P backend recovered LIO with a fixed restart-frame alignment");
    } else {
      last_keyframe_lio_local_ = lio_local;
    }

    last_keyframe_stamp_ = lio_msg.header.stamp;
    previous_encoder_velocity_ = wheel.velocity;
    if (period_event_inserted) {
      ROS_DEBUG("S2P admitted a structural-periodicity event at keyframe %zu", keyframe_);
    }
    return PosteriorPose{lio_msg.header.stamp, current_pose_, current_velocity_,
                         current_covariance_, keyframe_};
  }

  AisleFrame activeAisle() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return aisle_;
  }

  Mode mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
  }

 private:
  static Key X(std::size_t index) { return Symbol('x', index); }
  static Key V(std::size_t index) { return Symbol('v', index); }
  static Key C(std::size_t index) { return Symbol('c', index); }
  static Key L(std::size_t index) { return Symbol('l', index); }

  std::optional<PosteriorPose> initialize(const ros::Time& stamp,
                                          const Pose3& lio_local) {
    current_pose_ = Pose3(Rot3(), Point3(0.0, 0.0,
                                        config_.initial_body_height));
    current_velocity_ = Vector3::Zero();
    current_scales_ = Vector2::Ones();
    current_log_period_ = std::log(config_.initial_period);
    world_from_lio_ = current_pose_.compose(lio_local.inverse());
    last_keyframe_lio_local_ = lio_local;
    last_lio_stamp_ = stamp;
    last_keyframe_stamp_ = stamp;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    graph.add(gtsam::PriorFactor<Pose3>(
        X(0), current_pose_,
        gtsam::noiseModel::Diagonal::Sigmas(
            (Vector6() << 0.01, 0.01, 0.02, 0.02, 0.02, 0.02).finished())));
    graph.add(gtsam::PriorFactor<Vector3>(
        V(0), current_velocity_, gtsam::noiseModel::Isotropic::Sigma(3, 0.10)));
    graph.add(gtsam::PriorFactor<Vector2>(
        C(0), current_scales_, gtsam::noiseModel::Isotropic::Sigma(2, 0.05)));
    graph.add(gtsam::PriorFactor<double>(
        L(0), current_log_period_, gtsam::noiseModel::Isotropic::Sigma(1, 0.03)));
    values.insert(X(0), current_pose_);
    values.insert(V(0), current_velocity_);
    values.insert(C(0), current_scales_);
    values.insert(L(0), current_log_period_);
    isam_->update(graph, values);
    current_covariance_ = isam_->marginalCovariance(X(0));
    initialized_ = true;
    return PosteriorPose{stamp, current_pose_, current_velocity_, current_covariance_, 0};
  }

  static bool covariancePositiveDefinite(
      const boost::array<double, 36>& covariance) {
    Eigen::Matrix<double, 6, 6> matrix;
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 6; ++column) {
        matrix(row, column) = covariance[row * 6 + column];
      }
    }
    if (!matrix.array().isFinite().all()) return false;
    matrix = 0.5 * (matrix + matrix.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(matrix);
    return solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() > 1e-12;
  }

  Matrix3 predictedPositionCovariance(double dt) const {
    Matrix3 covariance = current_covariance_.block<3, 3>(3, 3);
    const double position_process = 0.02 + 0.05 * dt;
    covariance += Matrix3::Identity() * position_process * position_process;
    return covariance;
  }

  double railSubspaceStatistic(const Matrix3& covariance_w) const {
    if (!aisle_.valid) return 0.0;
    const Matrix3 r_aw = aisle_.r_wa.matrix().transpose();
    const Matrix3 covariance_a = r_aw * covariance_w * r_aw.transpose();
    Eigen::Matrix2d yz;
    yz << covariance_a(1, 1), covariance_a(1, 2), covariance_a(2, 1),
        covariance_a(2, 2);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(yz);
    return solver.info() == Eigen::Success ? solver.eigenvalues().maxCoeff()
                                           : std::numeric_limits<double>::infinity();
  }

  void updateAisleFrame(const Pose3& predicted_pose,
                        const FeatureBundle& features,
                        const WheelObservation& wheel, const ros::Time& stamp) {
    if (std::abs(wheel.yaw_rate) > 0.20) {
      if (turn_start_.isZero()) turn_start_ = stamp;
      if ((stamp - turn_start_).toSec() > 0.50) {
        aisle_.valid = false;
        aisle_.phase_initialized = false;
        last_period_index_ = 0;
      }
      return;
    }
    turn_start_ = ros::Time(0);

    if (!features.line.valid) return;
    const double line_heading_a = std::atan(features.line.coefficients.y());
    const Vector3 line_axis_w = features.line.frame.r_wa.rotate(
        Vector3(std::cos(line_heading_a), std::sin(line_heading_a), 0.0));
    const Vector3 line_axis_b = predicted_pose.rotation().unrotate(line_axis_w);
    const double line_heading_b = std::atan2(line_axis_b.y(), line_axis_b.x());

    double heading_world = features.line.frame.r_wa.yaw() + line_heading_a;
    if (features.vp.valid) {
      const Vector3 vp_axis_b = features.r_cb_corrected.inverse().rotate(
          features.vp.direction_camera.unitVector());
      const double vp_heading_b = std::atan2(vp_axis_b.y(), vp_axis_b.x());
      const double aligned_vp = alignUndirected(vp_heading_b, line_heading_b);
      const double disagreement = wrapPi(aligned_vp - line_heading_b);
      if (std::abs(disagreement) > config_.aisle_heading_agreement) {
        if (aisle_disagreement_start_.isZero()) {
          aisle_disagreement_start_ = stamp;
        } else if ((stamp - aisle_disagreement_start_).toSec() > 0.50) {
          aisle_.valid = false;
          aisle_.phase_initialized = false;
          last_period_index_ = 0;
        }
        return;
      }
      aisle_disagreement_start_ = ros::Time(0);
      const double fused_heading_b =
          line_heading_b + 0.5 * disagreement;
      heading_world = predicted_pose.rotation().yaw() + fused_heading_b;
    } else if (features.line.wall_fallback) {
      heading_world = alignUndirected(heading_world,
                                      predicted_pose.rotation().yaw());
      aisle_disagreement_start_ = ros::Time(0);
    } else {
      return;
    }
    if (!aisle_.valid) {
      aisle_.valid = true;
      aisle_.id = ++aisle_counter_;
      aisle_.r_wa = Rot3::Ypr(wrapPi(heading_world), 0.0, 0.0);
      aisle_.origin_w = predicted_pose.translation();
      aisle_.nominal_height = 0.0;
      aisle_.phase = 0.0;
      aisle_.phase_initialized = false;
      last_period_index_ = 0;
    }
  }

  bool straightAisleGate(const FeatureBundle& features,
                         const WheelObservation& wheel,
                         const Pose3& predicted_pose) const {
    if (!aisle_.valid || !features.vp.valid || !features.line.valid ||
        std::abs(wheel.yaw_rate) >= 0.20) {
      return false;
    }
    const Vector3 vp_axis_b = features.r_cb_corrected.inverse().rotate(
        features.vp.direction_camera.unitVector());
    const double vp_heading = std::atan2(vp_axis_b.y(), vp_axis_b.x());
    const double line_heading_a = std::atan(features.line.coefficients.y());
    const Vector3 line_axis_w = features.line.frame.r_wa.rotate(
        Vector3(std::cos(line_heading_a), std::sin(line_heading_a), 0.0));
    const Vector3 line_axis_b = predicted_pose.rotation().unrotate(line_axis_w);
    const double line_heading = std::atan2(line_axis_b.y(), line_axis_b.x());
    return std::abs(wrapPi(alignUndirected(vp_heading, line_heading) -
                           line_heading)) <= config_.aisle_heading_agreement;
  }

  Config config_;
  mutable std::mutex mutex_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  bool initialized_ = false;
  std::size_t keyframe_ = 0;
  Mode mode_ = Mode::kLio;
  Pose3 current_pose_;
  Vector3 current_velocity_ = Vector3::Zero();
  Vector2 current_scales_ = Vector2::Ones();
  double current_log_period_ = std::log(1.66);
  Matrix6 current_covariance_ = Matrix6::Identity();
  Pose3 world_from_lio_;
  Pose3 last_keyframe_lio_local_;
  ros::Time last_lio_stamp_;
  ros::Time last_keyframe_stamp_;
  double failure_duration_ = 0.0;
  double recovery_duration_ = 0.0;
  double previous_encoder_velocity_ = 0.0;
  AisleFrame aisle_;
  std::uint64_t aisle_counter_ = 0;
  std::int64_t last_period_index_ = 0;
  ros::Time last_period_lock_stamp_;
  ros::Time turn_start_;
  ros::Time aisle_disagreement_start_;
};

// ---------------------------------------------------------------------------
// Structure-aware RGB-D/LiDAR front end.
// ---------------------------------------------------------------------------

class StructureFrontend {
 public:
  struct ImageResult {
    cv::Mat structural_mask;
    VpObservation vp;
    FlowObservation flow;
    Rot3 r_cb_corrected;
  };

  struct LidarResult {
    GroundObservation ground;
    CenterlineObservation line;
    PeriodObservation period;
    Rot3 r_cb_corrected;
    bool valid = false;
  };

  explicit StructureFrontend(const Config& config)
      : config_(config),
        nominal_t_bc_(config.nominalTbc()),
        vp_filter_(4, 2, 0, CV_32F) {
    vp_filter_.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
    vp_filter_.measurementMatrix.at<float>(0, 0) = 1.0F;
    vp_filter_.measurementMatrix.at<float>(1, 1) = 1.0F;
    cv::setIdentity(vp_filter_.processNoiseCov, cv::Scalar::all(1e-3));
    cv::setIdentity(vp_filter_.measurementNoiseCov, cv::Scalar::all(4.0));
    cv::setIdentity(vp_filter_.errorCovPost, cv::Scalar::all(10.0));
    latest_r_cb_ = correctedRcb();
  }

  ImageResult processImages(const cv::Mat& rgb, const cv::Mat& depth,
                            const ros::Time& stamp,
                            const PosteriorPose& posterior,
                            const Pose3& motion_pose,
                            bool motion_pose_is_lio) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (motion_source_initialized_ &&
        motion_pose_is_lio != previous_motion_pose_is_lio_) {
      previous_gray_.release();
      previous_depth_.release();
      previous_image_stamp_ = ros::Time(0);
    }
    previous_motion_pose_is_lio_ = motion_pose_is_lio;
    motion_source_initialized_ = true;
    ImageResult result;
    result.r_cb_corrected = correctedRcb();
    if (rgb.empty() || depth.empty() || rgb.size() != depth.size()) {
      return result;
    }
    const cv::Mat enhanced = enhance(rgb);
    result.vp = detectVanishingPoint(enhanced, stamp);
    result.structural_mask = buildStructuralMask(enhanced, depth, posterior.pose,
                                                 result.vp);
    result.flow = estimateFlow(enhanced, depth, stamp, motion_pose,
                               result.r_cb_corrected);
    latest_mask_ = result.structural_mask.clone();
    latest_image_stamp_ = stamp;
    latest_vp_ = result.vp;
    latest_flow_ = result.flow;
    latest_r_cb_ = result.r_cb_corrected;
    return result;
  }

  LidarResult processLidar(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud_body,
                           const ros::Time& stamp,
                           const PosteriorPose& posterior,
                           const AisleFrame& aisle) {
    std::lock_guard<std::mutex> lock(mutex_);
    LidarResult result;
    if (!cloud_body || cloud_body->empty()) return result;
    result.valid = true;
    result.ground = fitGroundPlane(cloud_body, stamp, posterior.pose);
    const auto wall_heading = estimateWallHeading(cloud_body);
    VpObservation synchronized_vp = latest_vp_;
    if (!fresh(synchronized_vp.stamp, stamp, 0.25)) {
      synchronized_vp.valid = false;
    }
    double predicted_heading_body = wall_heading.heading_body;
    if (aisle.valid) {
      const Vector3 predicted_axis_body = posterior.pose.rotation().unrotate(
          aisle.axisWorld());
      predicted_heading_body = std::atan2(predicted_axis_body.y(),
                                           predicted_axis_body.x());
    }
    updateYawCorrection(wall_heading, synchronized_vp,
                        predicted_heading_body);
    result.r_cb_corrected = correctedRcb();
    result.line = fitCenterline(cloud_body, stamp, posterior.pose, aisle,
                                synchronized_vp, wall_heading,
                                result.r_cb_corrected);
    result.period = estimatePeriodicity(cloud_body, stamp, posterior.pose,
                                        aisle, result.r_cb_corrected);
    latest_ground_ = result.ground;
    latest_line_ = result.line;
    latest_period_ = result.period;
    latest_r_cb_ = result.r_cb_corrected;
    return result;
  }

  FeatureBundle bundleAt(const ros::Time& stamp) const {
    std::lock_guard<std::mutex> lock(mutex_);
    FeatureBundle bundle;
    bundle.r_cb_corrected = latest_r_cb_;
    if (fresh(latest_vp_.stamp, stamp, 0.25)) bundle.vp = latest_vp_;
    if (fresh(latest_flow_.stamp, stamp, 0.20)) bundle.flow = latest_flow_;
    if (fresh(latest_ground_.stamp, stamp, 0.25)) bundle.ground = latest_ground_;
    if (fresh(latest_line_.stamp, stamp, 0.25)) bundle.line = latest_line_;
    if (fresh(latest_period_.stamp, stamp, 0.35)) bundle.period = latest_period_;
    bundle.lidar_valid = bundle.ground.valid || bundle.line.valid ||
                         bundle.period.valid;
    return bundle;
  }

  cv::Mat latestMask() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_mask_.clone();
  }

 private:
  struct WallHeading {
    bool valid = false;
    double heading_body = 0.0;
    double center_normal_offset = 0.0;
    double confidence = 0.0;
    int support = 0;
  };

  static bool fresh(const ros::Time& source, const ros::Time& target,
                    const double tolerance) {
    return !source.isZero() && std::abs((target - source).toSec()) <= tolerance;
  }

  double depthMeters(const cv::Mat& depth, int row, int column) const {
    if (depth.type() == CV_16UC1) {
      return 0.001 * static_cast<double>(depth.at<std::uint16_t>(row, column));
    }
    if (depth.type() == CV_32FC1) {
      return static_cast<double>(depth.at<float>(row, column));
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

  static double flowDepthVariance(const double depth) {
    // Interpolated empirical profile reported for the retained camera/aisle
    // configuration: near-field mixing and far-field quantization receive
    // lower weights than the 1.8--2.4 m residual basin.
    constexpr std::array<double, 6> kDepth{
        0.30, 1.40, 1.80, 2.40, 3.20, 8.00};
    constexpr std::array<double, 6> kSigma{
        0.30, 0.30, 0.10, 0.10, 0.16, 0.30};
    if (depth <= kDepth.front()) return kSigma.front() * kSigma.front();
    if (depth >= kDepth.back()) return kSigma.back() * kSigma.back();
    for (std::size_t index = 1; index < kDepth.size(); ++index) {
      if (depth <= kDepth[index]) {
        const double fraction = (depth - kDepth[index - 1]) /
                                (kDepth[index] - kDepth[index - 1]);
        const double sigma = (1.0 - fraction) * kSigma[index - 1] +
                             fraction * kSigma[index];
        return sigma * sigma;
      }
    }
    return kSigma.back() * kSigma.back();
  }

  cv::Mat enhance(const cv::Mat& rgb) const {
    cv::Mat lookup(1, 256, CV_8U);
    for (int value = 0; value < 256; ++value) {
      lookup.at<std::uint8_t>(value) = cv::saturate_cast<std::uint8_t>(
          std::pow(value / 255.0, config_.gamma) * 255.0);
    }
    cv::Mat gamma_corrected;
    cv::LUT(rgb, lookup, gamma_corrected);
    cv::Mat lab;
    cv::cvtColor(gamma_corrected, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    cv::createCLAHE(3.0, cv::Size(8, 8))->apply(channels[0], channels[0]);
    cv::merge(channels, lab);
    cv::Mat output;
    cv::cvtColor(lab, output, cv::COLOR_Lab2BGR);
    return output;
  }

  VpObservation detectVanishingPoint(const cv::Mat& rgb,
                                     const ros::Time& stamp) {
    VpObservation observation;
    observation.stamp = stamp;
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 60.0, 160.0, 3, true);
    std::vector<cv::Vec4i> raw_lines;
    cv::HoughLinesP(edges, raw_lines, 1.0, CV_PI / 180.0, 45,
                    config_.min_image_line_length, 20.0);

    std::vector<cv::Vec3d> lines;
    lines.reserve(raw_lines.size());
    for (const auto& line : raw_lines) {
      const double dx = line[2] - line[0];
      const double dy = line[3] - line[1];
      const double length = std::hypot(dx, dy);
      if (length < config_.min_image_line_length) continue;
      const double angle = std::atan2(dy, dx);
      if (std::abs(std::sin(angle)) < 0.20) continue;
      double a = static_cast<double>(line[1] - line[3]);
      double b = static_cast<double>(line[2] - line[0]);
      double c = static_cast<double>(line[0] * line[3] - line[2] * line[1]);
      const double norm = std::hypot(a, b);
      if (norm < kEpsilon) continue;
      lines.emplace_back(a / norm, b / norm, c / norm);
    }
    if (lines.size() < 4) return observation;

    auto solve = [](const std::vector<cv::Vec3d>& equations,
                    cv::Point2d* point) -> bool {
      cv::Mat a(static_cast<int>(equations.size()), 2, CV_64F);
      cv::Mat b(static_cast<int>(equations.size()), 1, CV_64F);
      for (std::size_t i = 0; i < equations.size(); ++i) {
        a.at<double>(static_cast<int>(i), 0) = equations[i][0];
        a.at<double>(static_cast<int>(i), 1) = equations[i][1];
        b.at<double>(static_cast<int>(i), 0) = -equations[i][2];
      }
      const cv::SVD conditioning(a, cv::SVD::NO_UV);
      if (conditioning.w.total() < 2) return false;
      const double sigma_max = conditioning.w.at<double>(0, 0);
      const double sigma_min = conditioning.w.at<double>(1, 0);
      if (!std::isfinite(sigma_max) || !std::isfinite(sigma_min) ||
          sigma_max <= kEpsilon || sigma_min / sigma_max < 1e-3) {
        return false;
      }
      cv::Mat solution;
      if (!cv::solve(a, b, solution, cv::DECOMP_SVD)) return false;
      point->x = solution.at<double>(0);
      point->y = solution.at<double>(1);
      return std::isfinite(point->x) && std::isfinite(point->y);
    };

    cv::Point2d candidate;
    if (!solve(lines, &candidate)) return observation;
    std::vector<cv::Vec3d> inliers;
    for (const auto& line : lines) {
      if (std::abs(line[0] * candidate.x + line[1] * candidate.y + line[2]) <= 4.0) {
        inliers.push_back(line);
      }
    }
    if (inliers.size() < 4 || !solve(inliers, &candidate)) return observation;
    if (std::abs(candidate.x - config_.cx) > 0.33 * rgb.cols) return observation;

    const double dt = vp_last_stamp_.isZero() ? 1.0 / 30.0
                                               : clamp((stamp - vp_last_stamp_).toSec(), 1e-3, 0.2);
    vp_filter_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1.0F, 0.0F, static_cast<float>(dt), 0.0F,
        0.0F, 1.0F, 0.0F, static_cast<float>(dt),
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F);
    if (!vp_filter_initialized_) {
      vp_filter_.statePost = (cv::Mat_<float>(4, 1) <<
          static_cast<float>(candidate.x), static_cast<float>(candidate.y), 0.0F, 0.0F);
      vp_filter_initialized_ = true;
    } else {
      vp_filter_.predict();
      cv::Mat measurement = (cv::Mat_<float>(2, 1) <<
          static_cast<float>(candidate.x), static_cast<float>(candidate.y));
      vp_filter_.correct(measurement);
    }
    vp_last_stamp_ = stamp;
    observation.pixel = cv::Point2f(vp_filter_.statePost.at<float>(0),
                                    vp_filter_.statePost.at<float>(1));
    Vector3 direction((observation.pixel.x - config_.cx) / config_.fx,
                      (observation.pixel.y - config_.cy) / config_.fy, 1.0);
    direction.normalize();
    observation.direction_camera = Unit3(direction);
    observation.viewing_angle = std::acos(clamp(std::abs(direction.z()), 0.0, 1.0));
    observation.inliers = static_cast<int>(inliers.size());
    observation.confidence = clamp(static_cast<double>(inliers.size()) /
                                       std::max<std::size_t>(lines.size(), 1),
                                   0.0, 1.0);
    observation.valid = observation.viewing_angle <= 78.0 * kPi / 180.0;
    return observation;
  }

  cv::Mat buildStructuralMask(const cv::Mat& rgb, const cv::Mat& depth,
                              const Pose3& pose_wb,
                              const VpObservation& vp) const {
    cv::Mat hsv;
    cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);
    cv::Mat red_low, red_high, color_gate;
    cv::inRange(hsv, cv::Scalar(0, config_.hsv_s_min, config_.hsv_v_min),
                cv::Scalar(15, 255, 255), red_low);
    cv::inRange(hsv, cv::Scalar(165, config_.hsv_s_min, config_.hsv_v_min),
                cv::Scalar(180, 255, 255), red_high);
    cv::bitwise_or(red_low, red_high, color_gate);

    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8UC1);
    const Pose3 pose_wc = pose_wb.compose(nominal_t_bc_);
    for (int row = 0; row < depth.rows; ++row) {
      auto* output = mask.ptr<std::uint8_t>(row);
      const auto* color = color_gate.ptr<std::uint8_t>(row);
      for (int column = 0; column < depth.cols; ++column) {
        const double z = depthMeters(depth, row, column);
        if (!std::isfinite(z) || z < config_.depth_min || z > config_.depth_max) continue;
        const Point3 point_c((column - config_.cx) * z / config_.fx,
                             (row - config_.cy) * z / config_.fy, z);
        const double height = pose_wc.transformFrom(point_c).z();
        if (height <= config_.front_floor_height ||
            height >= config_.front_ceiling_height) {
          continue;
        }
        if (color[column] != 0) output[column] = 255;
      }
    }

    if (vp.valid) {
      const int margin = 5;
      std::vector<cv::Point> floor_polygon{
          cv::Point(cvRound(vp.pixel.x), cvRound(vp.pixel.y)),
          cv::Point(depth.cols - margin, depth.rows - 1),
          cv::Point(margin, depth.rows - 1)};
      cv::fillConvexPoly(mask, floor_polygon, cv::Scalar(0));
    }

    const int split = mask.rows / 2;
    cv::Mat far = mask(cv::Rect(0, 0, mask.cols, split));
    cv::Mat near = mask(cv::Rect(0, split, mask.cols, mask.rows - split));
    cv::morphologyEx(far, far, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT,
                         cv::Size(config_.morph_far_w, config_.morph_far_h)));
    cv::morphologyEx(near, near, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT,
                         cv::Size(config_.morph_near_w, config_.morph_near_h)));
    return mask;
  }

  FlowObservation estimateFlow(const cv::Mat& rgb, const cv::Mat& depth,
                               const ros::Time& stamp, const Pose3& pose_wb,
                               const Rot3& r_cb) {
    FlowObservation observation;
    observation.stamp = stamp;
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    if (previous_gray_.empty() || previous_depth_.empty() ||
        previous_image_stamp_.isZero()) {
      rememberImage(gray, depth, stamp, pose_wb);
      return observation;
    }
    const double dt = (stamp - previous_image_stamp_).toSec();
    if (dt <= 0.005 || dt > 0.20) {
      rememberImage(gray, depth, stamp, pose_wb);
      return observation;
    }

    std::vector<cv::Point2f> previous_points;
    cv::goodFeaturesToTrack(previous_gray_, previous_points, 500, 0.01, 5.0);
    if (previous_points.empty()) {
      rememberImage(gray, depth, stamp, pose_wb);
      return observation;
    }
    std::vector<cv::Point2f> current_points, reverse_points;
    std::vector<std::uint8_t> forward_status, reverse_status;
    std::vector<float> forward_error, reverse_error;
    cv::calcOpticalFlowPyrLK(previous_gray_, gray, previous_points,
                            current_points, forward_status, forward_error,
                            cv::Size(31, 31), 4);
    cv::calcOpticalFlowPyrLK(gray, previous_gray_, current_points,
                            reverse_points, reverse_status, reverse_error,
                            cv::Size(31, 31), 4);

    const Rot3 r_bc = r_cb.inverse();
    const Rot3 r_wc_previous = previous_pose_wb_.rotation().compose(r_bc);
    const Rot3 r_wc_current = pose_wb.rotation().compose(r_bc);
    const Rot3 r_current_previous = r_wc_current.inverse().compose(r_wc_previous);
    std::vector<double> velocities;
    std::vector<double> depths;
    for (std::size_t index = 0; index < previous_points.size(); ++index) {
      if (!forward_status[index] || !reverse_status[index] ||
          cv::norm(previous_points[index] - reverse_points[index]) > 1.0) {
        continue;
      }
      const int u = cvRound(previous_points[index].x);
      const int v = cvRound(previous_points[index].y);
      if (u < 0 || v < 0 || u >= previous_depth_.cols ||
          v >= previous_depth_.rows) {
        continue;
      }
      const double z = depthMeters(previous_depth_, v, u);
      if (!std::isfinite(z) || z < config_.depth_min || z > config_.depth_max) continue;
      Vector3 ray((previous_points[index].x - config_.cx) / config_.fx,
                  (previous_points[index].y - config_.cy) / config_.fy, 1.0);
      const Vector3 rotated = r_current_previous.rotate(ray);
      if (rotated.z() <= kEpsilon) continue;
      const double rotational_u = config_.fx * rotated.x() / rotated.z() + config_.cx;
      const double compensated_du = current_points[index].x - rotational_u;
      const double velocity_camera_x = -compensated_du * z / (config_.fx * dt);
      const Vector3 velocity_body = r_bc.rotate(Vector3(velocity_camera_x, 0.0, 0.0));
      if (std::isfinite(velocity_body.y())) {
        velocities.push_back(velocity_body.y());
        depths.push_back(z);
      }
    }
    observation.total = static_cast<int>(velocities.size());
    if (velocities.empty()) {
      rememberImage(gray, depth, stamp, pose_wb);
      return observation;
    }

    std::vector<std::size_t> best_inliers;
    const std::size_t trials = std::min<std::size_t>(velocities.size(), 64);
    for (std::size_t trial = 0; trial < trials; ++trial) {
      const std::size_t seed_index = trial * velocities.size() / trials;
      std::vector<std::size_t> inliers;
      for (std::size_t index = 0; index < velocities.size(); ++index) {
        if (std::abs(velocities[index] - velocities[seed_index]) <=
            config_.flow_ransac_threshold) {
          inliers.push_back(index);
        }
      }
      if (inliers.size() > best_inliers.size()) best_inliers.swap(inliers);
    }
    observation.inliers = static_cast<int>(best_inliers.size());
    const double instant_confidence = static_cast<double>(observation.inliers) /
                                      static_cast<double>(observation.total);
    flow_confidence_ = (1.0 - config_.flow_confidence_smoothing) *
                           flow_confidence_ +
                       config_.flow_confidence_smoothing *
                           instant_confidence;
    observation.confidence = flow_confidence_;
    if (observation.inliers < config_.flow_min_inliers) {
      rememberImage(gray, depth, stamp, pose_wb);
      return observation;
    }
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (const std::size_t index : best_inliers) {
      const double weight = 1.0 / (flowDepthVariance(depths[index]) + 1e-6);
      weighted_sum += weight * velocities[index];
      weight_sum += weight;
    }
    observation.lateral_velocity = weighted_sum / std::max(weight_sum, kEpsilon);
    observation.variance = config_.flow_sigma0 * config_.flow_sigma0 /
                           (flow_confidence_ + 1e-3);
    observation.valid = std::isfinite(observation.lateral_velocity) &&
                        observation.confidence > 0.0;
    rememberImage(gray, depth, stamp, pose_wb);
    return observation;
  }

  void rememberImage(const cv::Mat& gray, const cv::Mat& depth,
                     const ros::Time& stamp, const Pose3& pose_wb) {
    previous_gray_ = gray.clone();
    previous_depth_ = depth.clone();
    previous_image_stamp_ = stamp;
    previous_pose_wb_ = pose_wb;
  }

  GroundObservation fitGroundPlane(
      const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
      const ros::Time& stamp, const Pose3& pose_wb) const {
    GroundObservation observation;
    observation.stamp = stamp;
    pcl::PointCloud<pcl::PointXYZI>::Ptr candidates(
        new pcl::PointCloud<pcl::PointXYZI>());
    for (const auto& point : cloud->points) {
      if (std::isfinite(point.x) && std::isfinite(point.y) &&
          std::isfinite(point.z) && point.z > -1.5 && point.z < 0.35 &&
          std::hypot(point.x, point.y) < 8.0) {
        candidates->push_back(point);
      }
    }
    if (candidates->size() < 100) return observation;
    pcl::SACSegmentation<pcl::PointXYZI> segmentation;
    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setMaxIterations(config_.ground_ransac_iterations);
    segmentation.setDistanceThreshold(0.03);
    segmentation.setInputCloud(candidates);
    pcl::PointIndices inliers;
    pcl::ModelCoefficients coefficients;
    segmentation.segment(inliers, coefficients);
    if (inliers.indices.size() < 100 || coefficients.values.size() < 4) {
      return observation;
    }
    Vector3 normal(coefficients.values[0], coefficients.values[1],
                   coefficients.values[2]);
    const double norm = normal.norm();
    if (norm < kEpsilon) return observation;
    normal /= norm;
    double offset = coefficients.values[3] / norm;
    const Vector3 gravity_body = pose_wb.rotation().unrotate(Vector3(0.0, 0.0, -1.0));
    if (normal.dot(gravity_body) >= 0.0) {
      normal *= -1.0;
      offset *= -1.0;
    }
    if (normal.dot(-gravity_body) < std::cos(20.0 * kPi / 180.0)) {
      return observation;
    }
    observation.plane_body << normal, offset;
    observation.inliers = static_cast<int>(inliers.indices.size());
    observation.valid = true;
    return observation;
  }

  WallHeading estimateWallHeading(
      const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const {
    WallHeading result;
    pcl::PointCloud<pcl::PointXYZ>::Ptr wall(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<Vector2> left_support;
    std::vector<Vector2> right_support;
    for (const auto& point : cloud->points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z)) {
        continue;
      }
      if (point.x > 0.4 && point.x < 8.0 && std::abs(point.y) > 0.45 &&
          std::abs(point.y) < 2.5 && point.z > 0.15 && point.z < 2.5) {
        pcl::PointXYZ flattened;
        flattened.x = point.x;
        flattened.y = point.y;
        flattened.z = 0.0F;
        wall->push_back(flattened);
        if (point.y < 0.0F) {
          left_support.emplace_back(point.x, point.y);
        } else {
          right_support.emplace_back(point.x, point.y);
        }
      }
    }
    if (wall->size() < 30 || left_support.size() < 15 ||
        right_support.size() < 15) {
      return result;
    }
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(wall);
    const Eigen::Vector3f eigenvalues = pca.getEigenValues();
    if (eigenvalues.x() <= kEpsilon) return result;
    Eigen::Vector3f direction = pca.getEigenVectors().col(0);
    if (direction.x() < 0.0F) direction *= -1.0F;
    if (std::abs(direction.x()) < 0.5F) return result;
    result.heading_body = std::atan2(direction.y(), direction.x());
    const Vector2 normal(-std::sin(result.heading_body),
                         std::cos(result.heading_body));
    auto median_projection = [&normal](const std::vector<Vector2>& points) {
      std::vector<double> projections;
      projections.reserve(points.size());
      for (const Vector2& point : points) projections.push_back(normal.dot(point));
      const auto middle = projections.begin() +
                          static_cast<std::ptrdiff_t>(projections.size() / 2);
      std::nth_element(projections.begin(), middle, projections.end());
      return *middle;
    };
    const double left_offset = median_projection(left_support);
    const double right_offset = median_projection(right_support);
    if (std::abs(right_offset - left_offset) < 0.5) return result;
    result.center_normal_offset = 0.5 * (left_offset + right_offset);
    const double anisotropy = (eigenvalues.x() - eigenvalues.y()) /
                              std::max<double>(eigenvalues.x(), kEpsilon);
    const double balance = static_cast<double>(
        std::min(left_support.size(), right_support.size())) /
        static_cast<double>(std::max(left_support.size(), right_support.size()));
    result.confidence = clamp(anisotropy * balance, 0.0, 1.0);
    result.support = static_cast<int>(left_support.size() + right_support.size());
    result.valid = result.confidence >= config_.pca_linearity;
    return result;
  }

  void updateYawCorrection(const WallHeading& lidar,
                           const VpObservation& vp,
                           double predicted_heading_body) {
    if (!lidar.valid || !vp.valid || lidar.confidence < 0.4 ||
        vp.confidence < 0.2) {
      return;
    }
    // Form the raw cross-modal disparity with the nominal extrinsic.  The
    // current auxiliary correction is subtracted once in the innovation below;
    // using the corrected transform here would apply that subtraction twice.
    const Vector3 vp_body = nominal_t_bc_.rotation().rotate(
        vp.direction_camera.unitVector());
    const double vp_heading = std::atan2(vp_body.y(), vp_body.x());
    const double aligned_lidar = alignUndirected(lidar.heading_body,
                                                  predicted_heading_body);
    const double aligned_vp = alignUndirected(vp_heading,
                                               predicted_heading_body);
    const double disparity = wrapPi(aligned_vp - aligned_lidar);
    const double innovation = wrapPi(disparity - yaw_correction_);
    if (std::abs(innovation) <= config_.yaw_calib_safe) {
      yaw_correction_ = wrapPi(yaw_correction_ +
                               config_.yaw_calib_gain * innovation);
    }
  }

  Rot3 correctedRcb() const {
    const Rot3 nominal_r_cb = nominal_t_bc_.rotation().inverse();
    return nominal_r_cb.compose(Rot3::Rz(yaw_correction_));
  }

  bool projectBodyPoint(const Point3& point_b, const Rot3& r_cb,
                        cv::Point2i* pixel) const {
    const Pose3 nominal_t_cb = nominal_t_bc_.inverse();
    const Vector3 point_c = r_cb.rotate(point_b) + nominal_t_cb.translation();
    if (point_c.z() <= kEpsilon) return false;
    const int u = cvRound(config_.fx * point_c.x() / point_c.z() + config_.cx);
    const int v = cvRound(config_.fy * point_c.y() / point_c.z() + config_.cy);
    if (latest_mask_.empty() || u < 0 || v < 0 || u >= latest_mask_.cols ||
        v >= latest_mask_.rows) {
      return false;
    }
    *pixel = cv::Point2i(u, v);
    return true;
  }

  CenterlineObservation fitCenterline(
      const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
      const ros::Time& stamp, const Pose3& pose_wb, const AisleFrame& aisle,
      const VpObservation& vp, const WallHeading& wall_heading,
      const Rot3& r_cb) {
    CenterlineObservation observation;
    observation.stamp = stamp;
    // Before the first aisle lock, fit in a provisional gravity-aligned frame
    // whose longitudinal axis follows the current body heading.  This first
    // measurement can establish A(k); it is not inserted as a line factor until
    // the following observation carries the locked aisle id.
    AisleFrame fitting_frame = aisle;
    if (!fitting_frame.valid) {
      fitting_frame.valid = true;
      fitting_frame.id = 0;
      fitting_frame.r_wa = Rot3::Ypr(pose_wb.rotation().yaw(), 0.0, 0.0);
      fitting_frame.origin_w = pose_wb.translation();
    }
    observation.frame = fitting_frame;
    observation.aisle_id = fitting_frame.id;

    // When the image VP is unavailable, bilateral wall-slice PCA supplies a
    // fixed straight-line measurement.  The two points below lie on the
    // PCA-center line in the body plane; a rigid transform into A(k) preserves
    // linearity, so the fallback is the a=0 special case used by r_line.
    if (!vp.valid && wall_heading.valid) {
      const Vector2 axis_body(std::cos(wall_heading.heading_body),
                              std::sin(wall_heading.heading_body));
      const Vector2 normal_body(-axis_body.y(), axis_body.x());
      const Vector2 center_body =
          wall_heading.center_normal_offset * normal_body;
      const Point3 first_body(center_body.x(), center_body.y(), 0.0);
      const Point3 second_body(center_body.x() + 6.0 * axis_body.x(),
                               center_body.y() + 6.0 * axis_body.y(), 0.0);
      const Point3 first_aisle = fitting_frame.toAisle(
          pose_wb.transformFrom(first_body));
      const Point3 second_aisle = fitting_frame.toAisle(
          pose_wb.transformFrom(second_body));
      const double delta_x = second_aisle.x() - first_aisle.x();
      if (std::abs(delta_x) > 0.5) {
        const double slope = (second_aisle.y() - first_aisle.y()) / delta_x;
        const double intercept = first_aisle.y() - slope * first_aisle.x();
        observation.coefficients = Vector3(0.0, slope, intercept);
        observation.wall_fallback = true;
        observation.support = wall_heading.support;
        observation.lateral_sigma = clamp(
            0.08 / std::max(wall_heading.confidence, 0.1), 0.04, 0.15);
        observation.heading_sigma = clamp(
            0.06 / std::max(wall_heading.confidence, 0.1), 0.03, 0.12);
        observation.valid = observation.coefficients.array().isFinite().all();
        if (observation.valid) {
          smoothed_centerline_ = observation.coefficients;
          smoothed_centerline_aisle_ = fitting_frame.id;
          centerline_initialized_ = true;
          return observation;
        }
      }
    }
    if (!vp.valid) return observation;

    std::vector<Vector2> points_a;
    for (double x = 0.5; x <= 6.0; x += 0.5) {
      std::vector<double> left, right;
      for (const auto& point : cloud->points) {
        if (std::abs(point.x - x) > 0.20 || point.z < 0.15 || point.z > 2.5) continue;
        if (point.y < -0.35) left.push_back(point.y);
        if (point.y > 0.35) right.push_back(point.y);
      }
      if (left.empty() || right.empty()) continue;
      auto median = [](std::vector<double>* values) {
        const auto middle = values->begin() +
                            static_cast<std::ptrdiff_t>(values->size() / 2);
        std::nth_element(values->begin(), middle, values->end());
        return *middle;
      };
      const Point3 midpoint_b(x, 0.5 * (median(&left) + median(&right)), 0.0);
      const Point3 midpoint_a = fitting_frame.toAisle(pose_wb.transformFrom(midpoint_b));
      points_a.emplace_back(midpoint_a.x(), midpoint_a.y());
    }
    if (vp.valid) {
      const double slope = (vp.pixel.x - config_.cx) / config_.fx;
      const Rot3 r_bc = r_cb.inverse();
      for (double distance = 4.0; distance <= 12.0 + 1e-6; distance += 0.5) {
        const Vector3 pseudo_c(distance * slope, 0.0, distance);
        const Point3 pseudo_b(r_bc.rotate(pseudo_c) +
                              nominal_t_bc_.translation());
        const Point3 pseudo_a = fitting_frame.toAisle(pose_wb.transformFrom(pseudo_b));
        points_a.emplace_back(pseudo_a.x(), pseudo_a.y());
      }
    }
    if (points_a.size() < 5) return observation;
    Eigen::MatrixXd design(points_a.size(), 3);
    Eigen::VectorXd target(points_a.size());
    for (std::size_t i = 0; i < points_a.size(); ++i) {
      design(static_cast<Eigen::Index>(i), 0) = points_a[i].x() * points_a[i].x();
      design(static_cast<Eigen::Index>(i), 1) = points_a[i].x();
      design(static_cast<Eigen::Index>(i), 2) = 1.0;
      target(static_cast<Eigen::Index>(i)) = points_a[i].y();
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        design, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.rank() < 3) return observation;
    const Vector3 measured = svd.solve(target);
    if (smoothed_centerline_aisle_ != fitting_frame.id || !centerline_initialized_) {
      smoothed_centerline_ = measured;
      smoothed_centerline_aisle_ = fitting_frame.id;
      centerline_initialized_ = true;
    } else {
      smoothed_centerline_ = (1.0 - config_.centerline_smoothing) *
                                 smoothed_centerline_ +
                             config_.centerline_smoothing * measured;
    }
    observation.coefficients = smoothed_centerline_;
    observation.support = static_cast<int>(points_a.size());
    observation.lateral_sigma = clamp(0.25 /
                                          std::sqrt(static_cast<double>(points_a.size())),
                                      0.03, 0.12);
    observation.heading_sigma = clamp(0.15 /
                                          std::sqrt(static_cast<double>(points_a.size())),
                                      0.02, 0.08);
    observation.valid = observation.coefficients.array().isFinite().all();
    return observation;
  }

  static double otsuThreshold(const std::vector<float>& values) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    const auto minmax = std::minmax_element(values.begin(), values.end());
    if (*minmax.second - *minmax.first < 1e-6F) return *minmax.first;
    std::array<double, 256> histogram{};
    for (const float value : values) {
      const int bin = static_cast<int>(clamp(
          255.0 * (value - *minmax.first) / (*minmax.second - *minmax.first),
          0.0, 255.0));
      histogram[bin] += 1.0;
    }
    const double total = static_cast<double>(values.size());
    double sum = 0.0;
    for (int bin = 0; bin < 256; ++bin) sum += bin * histogram[bin];
    double background_weight = 0.0;
    double background_sum = 0.0;
    double best_variance = -1.0;
    int best_bin = 0;
    for (int bin = 0; bin < 256; ++bin) {
      background_weight += histogram[bin];
      if (background_weight <= 0.0) continue;
      const double foreground_weight = total - background_weight;
      if (foreground_weight <= 0.0) break;
      background_sum += bin * histogram[bin];
      const double mean_background = background_sum / background_weight;
      const double mean_foreground = (sum - background_sum) / foreground_weight;
      const double variance = background_weight * foreground_weight *
                              std::pow(mean_background - mean_foreground, 2);
      if (variance > best_variance) {
        best_variance = variance;
        best_bin = bin;
      }
    }
    return *minmax.first + (static_cast<double>(best_bin) / 255.0) *
                               (*minmax.second - *minmax.first);
  }

  PeriodObservation estimatePeriodicity(
      const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
      const ros::Time& stamp, const Pose3& pose_wb, const AisleFrame& aisle,
      const Rot3& r_cb) {
    PeriodObservation observation;
    observation.stamp = stamp;
    observation.spacing = period_initialized_ ? smoothed_period_ : config_.initial_period;
    if (!aisle.valid) return observation;

    std::vector<std::vector<float>> sector_intensity(config_.radial_sectors);
    std::vector<int> sectors(cloud->size(), 0);
    for (std::size_t index = 0; index < cloud->size(); ++index) {
      const auto& point = cloud->points[index];
      const double angle = std::atan2(point.y, point.x);
      const int sector = std::min(config_.radial_sectors - 1,
                                  std::max(0, static_cast<int>(
                                      (angle + kPi) / (2.0 * kPi) *
                                      config_.radial_sectors)));
      sectors[index] = sector;
      sector_intensity[sector].push_back(point.intensity);
    }
    std::vector<double> thresholds(config_.radial_sectors);
    for (int sector = 0; sector < config_.radial_sectors; ++sector) {
      thresholds[sector] = otsuThreshold(sector_intensity[sector]);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr candidates(
        new pcl::PointCloud<pcl::PointXYZI>());
    for (std::size_t index = 0; index < cloud->size(); ++index) {
      const auto& point = cloud->points[index];
      if (point.intensity >= thresholds[sectors[index]] && point.x > 0.2 &&
          point.x < 12.0 && std::abs(point.y) < 3.0 && point.z > 0.05 &&
          point.z < 2.8) {
        candidates->push_back(point);
      }
    }
    if (candidates->size() < 3) return observation;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZI>());
    tree->setInputCloud(candidates);
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> clustering;
    clustering.setClusterTolerance(0.15);
    clustering.setMinClusterSize(1);
    clustering.setMaxClusterSize(1000);
    clustering.setSearchMethod(tree);
    clustering.setInputCloud(candidates);
    std::vector<pcl::PointIndices> clusters;
    clustering.extract(clusters);

    std::vector<double> support_x;
    double nearest_support_distance = std::numeric_limits<double>::infinity();
    const double robot_x = aisle.toAisle(pose_wb.translation()).x();
    for (const auto& cluster : clusters) {
      if (cluster.indices.empty()) continue;
      double min_y = std::numeric_limits<double>::infinity();
      double max_y = -std::numeric_limits<double>::infinity();
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*candidates, cluster.indices, centroid);
      for (const int index : cluster.indices) {
        min_y = std::min<double>(min_y, candidates->points[index].y);
        max_y = std::max<double>(max_y, candidates->points[index].y);
      }
      const Point3 centroid_b(centroid.x(), centroid.y(), centroid.z());
      cv::Point2i pixel;
      const bool projection_valid = fresh(latest_image_stamp_, stamp, 0.25) &&
                                    projectBodyPoint(centroid_b, r_cb, &pixel);
      const bool visually_confirmed = !config_.enable_semantic_gate ||
          (projection_valid &&
           latest_mask_.at<std::uint8_t>(pixel.y, pixel.x) != 0);
      const int count = static_cast<int>(cluster.indices.size());
      const double lateral_span = max_y - min_y;
      const bool geometry_supported =
          (count >= 2 && lateral_span > config_.structural_span) ||
          (count >= 3 && lateral_span > config_.structural_span_dense);
      if (!visually_confirmed && !geometry_supported) continue;

      if (count >= 3) {
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*candidates, cluster.indices,
                                               centroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        if (solver.info() != Eigen::Success) continue;
        const auto eigenvalues = solver.eigenvalues();
        const double linearity = (eigenvalues(2) - eigenvalues(1)) /
                                 std::max<double>(eigenvalues(2), kEpsilon);
        if (linearity < config_.pca_linearity) continue;
      }
      const Point3 point_a = aisle.toAisle(pose_wb.transformFrom(centroid_b));
      support_x.push_back(point_a.x());
      nearest_support_distance = std::min(nearest_support_distance,
                                          std::abs(point_a.x() - robot_x));
    }
    observation.support = static_cast<int>(support_x.size());
    observation.semantic_gate = observation.support >= config_.period_support_min;
    if (!observation.semantic_gate) return observation;

    const auto minmax = std::minmax_element(support_x.begin(), support_x.end());
    const int bins = std::max(1, static_cast<int>(
        std::ceil((*minmax.second - *minmax.first) / config_.histogram_resolution)) + 1);
    std::vector<double> histogram(bins, 0.0);
    for (const double x : support_x) {
      const int bin = std::min(bins - 1, std::max(0, static_cast<int>(
          std::lround((x - *minmax.first) / config_.histogram_resolution))));
      histogram[bin] += 1.0;
    }
    const double zero_lag = std::inner_product(histogram.begin(), histogram.end(),
                                               histogram.begin(), 0.0);
    if (zero_lag <= kEpsilon) return observation;
    const int min_lag = static_cast<int>(std::ceil(config_.period_min /
                                                   config_.histogram_resolution));
    const int max_lag = static_cast<int>(std::floor(config_.period_max /
                                                    config_.histogram_resolution));
    std::vector<double> correlation(max_lag + 1, 0.0);
    int peak_lag = -1;
    double peak = -std::numeric_limits<double>::infinity();
    double second_peak = -std::numeric_limits<double>::infinity();
    for (int lag = min_lag; lag <= max_lag && lag < bins; ++lag) {
      for (int index = 0; index + lag < bins; ++index) {
        correlation[lag] += histogram[index] * histogram[index + lag];
      }
      if (correlation[lag] > peak) {
        second_peak = peak;
        peak = correlation[lag];
        peak_lag = lag;
      } else if (correlation[lag] > second_peak) {
        second_peak = correlation[lag];
      }
    }
    if (peak_lag < 0 || peak <= 0.0 ||
        (std::isfinite(second_peak) && peak <= second_peak + kEpsilon)) {
      return observation;
    }
    double sub_bin = 0.0;
    if (peak_lag > min_lag && peak_lag < max_lag &&
        peak_lag + 1 < bins) {
      const double left = correlation[peak_lag - 1];
      const double center = correlation[peak_lag];
      const double right = correlation[peak_lag + 1];
      const double denominator = left - 2.0 * center + right;
      if (std::abs(denominator) > kEpsilon) {
        sub_bin = clamp(0.5 * (left - right) / denominator, -0.5, 0.5);
      }
    }
    const double measured_period = (peak_lag + sub_bin) *
                                   config_.histogram_resolution;
    observation.confidence = clamp(peak / zero_lag, 0.0, 1.0);
    if (measured_period < config_.period_min || measured_period > config_.period_max ||
        observation.confidence < config_.period_confidence_min) {
      return observation;
    }
    if (!period_initialized_ || period_aisle_id_ != aisle.id) {
      smoothed_period_ = measured_period;
      period_initialized_ = true;
      period_aisle_id_ = aisle.id;
      last_frontend_period_index_.reset();
    } else {
      smoothed_period_ = (1.0 - config_.period_smoothing) * smoothed_period_ +
                         config_.period_smoothing * measured_period;
    }
    observation.spacing = smoothed_period_;
    observation.coherent_measurement = true;
    observation.valid = true;
    const std::int64_t current_index = static_cast<std::int64_t>(std::llround(
        robot_x / std::max(smoothed_period_, 1e-6)));
    const bool new_index = !last_frontend_period_index_.has_value() ||
                           current_index != *last_frontend_period_index_;
    observation.phase_gate = nearest_support_distance <= 0.20;
    observation.event = new_index && observation.phase_gate;
    if (observation.event) last_frontend_period_index_ = current_index;
    return observation;
  }

  Config config_;
  Pose3 nominal_t_bc_;
  mutable std::mutex mutex_;
  cv::KalmanFilter vp_filter_;
  bool vp_filter_initialized_ = false;
  ros::Time vp_last_stamp_;
  cv::Mat previous_gray_;
  cv::Mat previous_depth_;
  ros::Time previous_image_stamp_;
  Pose3 previous_pose_wb_;
  bool motion_source_initialized_ = false;
  bool previous_motion_pose_is_lio_ = false;
  double flow_confidence_ = 0.0;
  double yaw_correction_ = 0.0;
  cv::Mat latest_mask_;
  ros::Time latest_image_stamp_;
  VpObservation latest_vp_;
  FlowObservation latest_flow_;
  GroundObservation latest_ground_;
  CenterlineObservation latest_line_;
  PeriodObservation latest_period_;
  Rot3 latest_r_cb_;
  bool centerline_initialized_ = false;
  std::uint64_t smoothed_centerline_aisle_ = 0;
  Vector3 smoothed_centerline_ = Vector3::Zero();
  bool period_initialized_ = false;
  std::uint64_t period_aisle_id_ = 0;
  double smoothed_period_ = 1.66;
  std::optional<std::int64_t> last_frontend_period_index_;
};

// ---------------------------------------------------------------------------
// Downstream spatiotemporal semantic mapper (STSM).
// ---------------------------------------------------------------------------

class SpatiotemporalSemanticMapper {
 public:
  enum class SemanticClass : std::uint8_t { kGround, kStructure, kObstacle };

  SpatiotemporalSemanticMapper(const Config& config, ros::NodeHandle& nh)
      : config_(config), tree_(config.voxel_resolution),
        nominal_t_bc_(config.nominalTbc()) {
    tree_.setProbHit(config_.occupancy_hit);
    tree_.setProbMiss(config_.occupancy_miss);
    tree_.setClampingThresMin(config_.occupancy_min);
    tree_.setClampingThresMax(config_.occupancy_max);
    tree_.setOccupancyThres(config_.occupancy_threshold);
    octomap_publisher_ = nh.advertise<octomap_msgs::Octomap>("semantic_octomap", 1, true);
    marker_publisher_ = nh.advertise<visualization_msgs::MarkerArray>(
        "semantic_octomap_markers", 1, true);
    grid_publisher_ = nh.advertise<nav_msgs::OccupancyGrid>("semantic_map_2d", 1, true);
  }

  void processPacket(const cv::Mat& depth, const cv::Mat& structural_mask,
                     const ros::Time& stamp, const PosteriorPose& posterior,
                     std::uint64_t packet_id) {
    if (!config_.enable_stsm || depth.empty() || structural_mask.empty()) return;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!processed_packets_.insert(packet_id).second) return;
    const Pose3 pose_wc = posterior.pose.compose(nominal_t_bc_);
    const octomap::point3d origin(static_cast<float>(pose_wc.x()),
                                  static_cast<float>(pose_wc.y()),
                                  static_cast<float>(pose_wc.z()));
    std::unordered_map<octomap::OcTreeKey, BufferedRay,
                       octomap::OcTreeKey::KeyHash> packet_rays;
    for (int row = 0; row < depth.rows; row += config_.map_pixel_stride) {
      for (int column = 0; column < depth.cols; column += config_.map_pixel_stride) {
        const double z = depthMeters(depth, row, column);
        if (!std::isfinite(z) || z < config_.depth_min || z > config_.depth_max) continue;
        const Point3 point_c((column - config_.cx) * z / config_.fx,
                             (row - config_.cy) * z / config_.fy, z);
        const Point3 point_w = pose_wc.transformFrom(point_c);
        const octomap::point3d endpoint(static_cast<float>(point_w.x()),
                                        static_cast<float>(point_w.y()),
                                        static_cast<float>(point_w.z()));
        octomap::OcTreeKey key;
        if (!tree_.coordToKeyChecked(endpoint, key)) continue;
        SemanticClass label = SemanticClass::kObstacle;
        if (std::abs(point_w.z()) < config_.map_ground_height) {
          label = SemanticClass::kGround;
        } else if (structural_mask.at<std::uint8_t>(row, column) != 0) {
          label = SemanticClass::kStructure;
        }
        const int support = localSupport(depth, structural_mask, row, column, z,
                                         label);
        auto candidate = packet_rays.find(key);
        if (candidate == packet_rays.end() ||
            support > candidate->second.cluster_support) {
          packet_rays[key] = BufferedRay{origin, endpoint, label, stamp, packet_id,
                                         support};
        }
      }
    }

    // Non-floor persistence uses metric Euclidean support.  The image-local
    // count above is retained only as a cheap seed; this clustering result is
    // the value used by the promotion gate.
    annotateEuclideanClusterSupport(&packet_rays);

    for (const auto& item : packet_rays) {
      const auto& key = item.first;
      const auto& ray = item.second;
      if (promoted_voxels_.count(key) != 0) {
        auto& last_packet = promoted_last_packet_[key];
        if (last_packet != packet_id) {
          integrateRay(ray);
          last_packet = packet_id;
        }
        continue;
      }
      CandidateRecord& record = candidates_[key];
      if (record.hits == 0) {
        record.first = stamp;
        record.last_view = ray.origin;
        record.views = 1;
      } else if ((ray.origin - record.last_view).norm() >=
                 config_.viewpoint_baseline) {
        ++record.views;
        record.last_view = ray.origin;
      }
      ++record.hits;
      record.last = stamp;
      record.max_cluster_support = std::max(record.max_cluster_support,
                                            ray.cluster_support);
      record.rays.push_back(ray);
      // Per-packet voxel deduplication is the compression step.  Keep every
      // distinct packet ray until promotion so replay remains complete.
      increment(record.pending_counts, ray.label);

      const bool persistence_gate = record.hits >= config_.persistence_hits &&
                                    record.views >= config_.persistence_views &&
                                    (record.last - record.first).toSec() >=
                                        config_.persistence_time;
      const bool cluster_gate = ray.label == SemanticClass::kGround ||
                                record.max_cluster_support >=
                                    config_.nonfloor_cluster_min;
      if (persistence_gate && cluster_gate) promote(key, record);
    }
    expireCandidates(stamp);
    tree_.updateInnerOccupancy();
    if (++packets_since_publish_ >= 10) {
      packets_since_publish_ = 0;
      publish(stamp);
    }
  }

  void publishNow(const ros::Time& stamp) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    publish(stamp);
  }

 private:
  struct Counts {
    int ground = 0;
    int structure = 0;
    int obstacle = 0;
  };

  struct BufferedRay {
    octomap::point3d origin;
    octomap::point3d endpoint;
    SemanticClass label = SemanticClass::kObstacle;
    ros::Time stamp;
    std::uint64_t packet_id = 0;
    int cluster_support = 0;
  };

  struct CandidateRecord {
    int hits = 0;
    int views = 0;
    int max_cluster_support = 0;
    octomap::point3d last_view;
    ros::Time first;
    ros::Time last;
    std::deque<BufferedRay> rays;
    Counts pending_counts;
  };

  using PacketRayMap = std::unordered_map<octomap::OcTreeKey, BufferedRay,
                                          octomap::OcTreeKey::KeyHash>;

  double depthMeters(const cv::Mat& depth, int row, int column) const {
    if (depth.type() == CV_16UC1) {
      return 0.001 * static_cast<double>(depth.at<std::uint16_t>(row, column));
    }
    if (depth.type() == CV_32FC1) {
      return static_cast<double>(depth.at<float>(row, column));
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

  int localSupport(const cv::Mat& depth, const cv::Mat& mask, int row,
                   int column, double center_depth, SemanticClass label) const {
    int support = 0;
    for (int dv = -2; dv <= 2; ++dv) {
      for (int du = -2; du <= 2; ++du) {
        const int v = row + dv;
        const int u = column + du;
        if (v < 0 || u < 0 || v >= depth.rows || u >= depth.cols) continue;
        const double neighbor_depth = depthMeters(depth, v, u);
        if (!std::isfinite(neighbor_depth) ||
            std::abs(neighbor_depth - center_depth) > 0.10) {
          continue;
        }
        if (label != SemanticClass::kStructure ||
            mask.at<std::uint8_t>(v, u) != 0) {
          ++support;
        }
      }
    }
    return support;
  }

  void annotateEuclideanClusterSupport(PacketRayMap* packet_rays) const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr endpoints(
        new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<octomap::OcTreeKey> keys;
    endpoints->reserve(packet_rays->size());
    keys.reserve(packet_rays->size());
    for (const auto& item : *packet_rays) {
      if (item.second.label == SemanticClass::kGround) continue;
      pcl::PointXYZ point;
      point.x = item.second.endpoint.x();
      point.y = item.second.endpoint.y();
      point.z = item.second.endpoint.z();
      endpoints->push_back(point);
      keys.push_back(item.first);
    }
    if (endpoints->empty()) return;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(endpoints);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> clustering;
    clustering.setClusterTolerance(0.15);
    clustering.setMinClusterSize(1);
    clustering.setMaxClusterSize(static_cast<int>(endpoints->size()));
    clustering.setSearchMethod(tree);
    clustering.setInputCloud(endpoints);
    std::vector<pcl::PointIndices> clusters;
    clustering.extract(clusters);
    for (const auto& cluster : clusters) {
      const int support = static_cast<int>(cluster.indices.size());
      for (const int index : cluster.indices) {
        if (index >= 0 && static_cast<std::size_t>(index) < keys.size()) {
          (*packet_rays)[keys[static_cast<std::size_t>(index)]].cluster_support =
              support;
        }
      }
    }
  }

  static void increment(Counts& counts, SemanticClass label) {
    if (label == SemanticClass::kGround) ++counts.ground;
    if (label == SemanticClass::kStructure) ++counts.structure;
    if (label == SemanticClass::kObstacle) ++counts.obstacle;
  }

  void promote(const octomap::OcTreeKey& key,
               const CandidateRecord& record) {
    std::vector<BufferedRay> ordered(record.rays.begin(), record.rays.end());
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const BufferedRay& left, const BufferedRay& right) {
                       return left.stamp < right.stamp;
                     });
    std::uint64_t last_packet = std::numeric_limits<std::uint64_t>::max();
    for (const auto& ray : ordered) {
      if (ray.packet_id == last_packet) continue;
      integrateRay(ray);
      last_packet = ray.packet_id;
    }
    promoted_voxels_.insert(key);
    promoted_last_packet_[key] = last_packet;
    candidates_.erase(key);
  }

  void integrateRay(const BufferedRay& ray) {
    octomap::KeyRay free_cells;
    if (tree_.computeRayKeys(ray.origin, ray.endpoint, free_cells)) {
      for (const auto& key : free_cells) tree_.updateNode(key, false, true);
    }
    octomap::ColorOcTreeNode* endpoint_node = tree_.updateNode(ray.endpoint, true, true);
    octomap::OcTreeKey endpoint_key;
    if (!tree_.coordToKeyChecked(ray.endpoint, endpoint_key)) return;
    increment(semantic_counts_[endpoint_key], ray.label);
    if (endpoint_node && tree_.isNodeOccupied(endpoint_node)) {
      setNodeColor(endpoint_key, endpoint_node);
    }
  }

  SemanticClass classify(const octomap::OcTreeKey& key,
                         const octomap::ColorOcTreeNode& node) const {
    const octomap::point3d center = tree_.keyToCoord(key);
    if (std::abs(center.z()) < config_.map_ground_height) {
      return SemanticClass::kGround;
    }
    const auto found = semantic_counts_.find(key);
    if (found != semantic_counts_.end()) {
      const Counts& counts = found->second;
      const int total = counts.ground + counts.structure + counts.obstacle;
      const double structure_probability = total > 0
                                               ? static_cast<double>(counts.structure) / total
                                               : 0.0;
      if (structure_probability > config_.semantic_threshold) {
        return SemanticClass::kStructure;
      }
    }
    (void)node;
    return SemanticClass::kObstacle;
  }

  void setNodeColor(const octomap::OcTreeKey& key,
                    octomap::ColorOcTreeNode* node) {
    if (!node || !tree_.isNodeOccupied(node)) return;
    const SemanticClass label = classify(key, *node);
    if (label == SemanticClass::kGround) node->setColor(50, 170, 70);
    if (label == SemanticClass::kStructure) node->setColor(210, 55, 45);
    if (label == SemanticClass::kObstacle) node->setColor(135, 135, 135);
  }

  int occupiedNeighbors(const octomap::OcTreeKey& key) const {
    const octomap::point3d center = tree_.keyToCoord(key);
    int count = 0;
    const double resolution = tree_.getResolution();
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          const octomap::point3d neighbor(
              center.x() + static_cast<float>(dx * resolution),
              center.y() + static_cast<float>(dy * resolution),
              center.z() + static_cast<float>(dz * resolution));
          const auto* node = tree_.search(neighbor);
          if (node && tree_.isNodeOccupied(node)) ++count;
        }
      }
    }
    return count;
  }

  void expireCandidates(const ros::Time& now) {
    for (auto iterator = candidates_.begin(); iterator != candidates_.end();) {
      if ((now - iterator->second.last).toSec() > config_.candidate_decay) {
        iterator = candidates_.erase(iterator);
      } else {
        ++iterator;
      }
    }
    if (processed_packets_.size() > 100000) processed_packets_.clear();
  }

  void publish(const ros::Time& stamp) {
    octomap::ColorOcTree final_tree(config_.voxel_resolution);
    final_tree.setProbHit(config_.occupancy_hit);
    final_tree.setProbMiss(config_.occupancy_miss);
    final_tree.setOccupancyThres(config_.occupancy_threshold);
    visualization_msgs::Marker marker;
    marker.header.frame_id = config_.world_frame;
    marker.header.stamp = stamp;
    marker.ns = "stsm_final";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = config_.voxel_resolution;
    marker.scale.y = config_.voxel_resolution;
    marker.scale.z = config_.voxel_resolution;

    for (auto iterator = tree_.begin_leafs(); iterator != tree_.end_leafs(); ++iterator) {
      if (!tree_.isNodeOccupied(*iterator)) continue;
      const octomap::OcTreeKey key = iterator.getKey();
      if (occupiedNeighbors(key) < config_.occupied_neighbor_min) continue;
      const octomap::point3d center = iterator.getCoordinate();
      auto* final_node = final_tree.updateNode(center, true);
      if (!final_node) continue;
      setNodeColor(key, &(*iterator));
      const auto color = iterator->getColor();
      final_node->setColor(color.r, color.g, color.b);
      geometry_msgs::Point point;
      point.x = center.x();
      point.y = center.y();
      point.z = center.z();
      marker.points.push_back(point);
      std_msgs::ColorRGBA marker_color;
      marker_color.r = color.r / 255.0F;
      marker_color.g = color.g / 255.0F;
      marker_color.b = color.b / 255.0F;
      marker_color.a = 1.0F;
      marker.colors.push_back(marker_color);
    }
    final_tree.updateInnerOccupancy();
    octomap_msgs::Octomap message;
    message.header.frame_id = config_.world_frame;
    message.header.stamp = stamp;
    if (octomap_msgs::fullMapToMsg(final_tree, message)) {
      octomap_publisher_.publish(message);
    }
    visualization_msgs::MarkerArray markers;
    markers.markers.push_back(marker);
    marker_publisher_.publish(markers);
    publishGrid(final_tree, stamp);
  }

  void publishGrid(const octomap::ColorOcTree& tree, const ros::Time& stamp) {
    double min_x, min_y, min_z, max_x, max_y, max_z;
    tree.getMetricMin(min_x, min_y, min_z);
    tree.getMetricMax(max_x, max_y, max_z);
    (void)min_z;
    (void)max_z;
    if (!std::isfinite(min_x) || max_x <= min_x || max_y <= min_y) return;
    nav_msgs::OccupancyGrid grid;
    grid.header.frame_id = config_.world_frame;
    grid.header.stamp = stamp;
    grid.info.resolution = config_.voxel_resolution;
    grid.info.width = static_cast<std::uint32_t>(std::ceil((max_x - min_x) /
                                                          config_.voxel_resolution)) + 1;
    grid.info.height = static_cast<std::uint32_t>(std::ceil((max_y - min_y) /
                                                           config_.voxel_resolution)) + 1;
    grid.info.origin.position.x = min_x;
    grid.info.origin.position.y = min_y;
    grid.info.origin.orientation.w = 1.0;
    grid.data.assign(grid.info.width * grid.info.height, -1);
    for (auto iterator = tree.begin_leafs(); iterator != tree.end_leafs(); ++iterator) {
      if (!tree.isNodeOccupied(*iterator)) continue;
      const int x = static_cast<int>((iterator.getX() - min_x) /
                                     config_.voxel_resolution);
      const int y = static_cast<int>((iterator.getY() - min_y) /
                                     config_.voxel_resolution);
      if (x >= 0 && y >= 0 && x < static_cast<int>(grid.info.width) &&
          y < static_cast<int>(grid.info.height)) {
        grid.data[y * grid.info.width + x] = 100;
      }
    }
    grid_publisher_.publish(grid);
  }

  Config config_;
  octomap::ColorOcTree tree_;
  Pose3 nominal_t_bc_;
  mutable std::shared_mutex mutex_;
  std::unordered_map<octomap::OcTreeKey, CandidateRecord,
                     octomap::OcTreeKey::KeyHash> candidates_;
  std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>
      promoted_voxels_;
  std::unordered_map<octomap::OcTreeKey, std::uint64_t,
                     octomap::OcTreeKey::KeyHash> promoted_last_packet_;
  std::unordered_map<octomap::OcTreeKey, Counts,
                     octomap::OcTreeKey::KeyHash> semantic_counts_;
  std::unordered_set<std::uint64_t> processed_packets_;
  int packets_since_publish_ = 0;
  ros::Publisher octomap_publisher_;
  ros::Publisher marker_publisher_;
  ros::Publisher grid_publisher_;
};

class TimedPoseBuffer {
 public:
  void add(const ros::Time& stamp, const Pose3& pose) {
    std::lock_guard<std::mutex> lock(mutex_);
    poses_.emplace_back(stamp, pose);
    while (poses_.size() > 400) poses_.pop_front();
  }

  std::optional<Pose3> nearest(const ros::Time& stamp,
                               double tolerance = 0.20) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (poses_.empty()) return std::nullopt;
    const auto* best = &poses_.front();
    double best_difference = std::abs((best->first - stamp).toSec());
    for (const auto& item : poses_) {
      const double difference = std::abs((item.first - stamp).toSec());
      if (difference < best_difference) {
        best = &item;
        best_difference = difference;
      }
    }
    return best_difference <= tolerance ? std::optional<Pose3>(best->second)
                                        : std::nullopt;
  }

 private:
  mutable std::mutex mutex_;
  std::deque<std::pair<ros::Time, Pose3>> poses_;
};

class PosteriorBuffer {
 public:
  void add(const PosteriorPose& posterior) {
    std::lock_guard<std::mutex> lock(mutex_);
    poses_.push_back(posterior);
    while (poses_.size() > 400) poses_.pop_front();
  }

  std::optional<PosteriorPose> nearest(const ros::Time& stamp,
                                       double tolerance = 0.20) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (poses_.empty()) return std::nullopt;
    const PosteriorPose* best = &poses_.front();
    double best_difference = std::abs((best->stamp - stamp).toSec());
    for (const auto& item : poses_) {
      const double difference = std::abs((item.stamp - stamp).toSec());
      if (difference < best_difference) {
        best = &item;
        best_difference = difference;
      }
    }
    return best_difference <= tolerance
               ? std::optional<PosteriorPose>(*best)
               : std::nullopt;
  }

 private:
  mutable std::mutex mutex_;
  std::deque<PosteriorPose> poses_;
};

class RunLogger {
 public:
  RunLogger() {
    const char* directory = std::getenv("S2P_LOG_DIR");
    if (!directory || std::string(directory).empty()) return;
    std::string path(directory);
    if (path.back() != '/' && path.back() != '\\') path += '/';
    trajectory_.open(path + "trajectory.txt", std::ios::out);
    if (trajectory_) {
      trajectory_ << "time x y z qx qy qz qw vx vy vz keyframe mode\n";
    }
  }

  void log(const PosteriorPose& posterior, FactorGraphBackend::Mode mode) {
    if (!trajectory_) return;
    const auto quaternion = posterior.pose.rotation().toQuaternion();
    trajectory_ << std::fixed << std::setprecision(9)
                << posterior.stamp.toSec() << ' ' << posterior.pose.x() << ' '
                << posterior.pose.y() << ' ' << posterior.pose.z() << ' '
                << quaternion.x() << ' ' << quaternion.y() << ' '
                << quaternion.z() << ' ' << quaternion.w() << ' '
                << posterior.velocity.x() << ' ' << posterior.velocity.y() << ' '
                << posterior.velocity.z() << ' ' << posterior.keyframe << ' '
                << (mode == FactorGraphBackend::Mode::kLio ? "lio" : "fallback")
                << '\n';
  }

 private:
  std::ofstream trajectory_;
};

class S2pSlamNode {
 public:
  explicit S2pSlamNode(ros::NodeHandle& nh_private)
      : nh_(),
        nh_private_(nh_private),
        config_(readConfig(nh_private_)),
        backend_(config_),
        frontend_(config_),
        mapper_(config_, nh_private_) {
    lio_subscriber_ = nh_.subscribe(config_.lio_odom_topic, 100,
                                    &S2pSlamNode::lioCallback, this);
    wheel_subscriber_ = nh_.subscribe(config_.wheel_odom_topic, 100,
                                      &S2pSlamNode::wheelCallback, this);
    command_subscriber_ = nh_.subscribe(config_.command_topic, 100,
                                        &S2pSlamNode::commandCallback, this);
    lio_status_subscriber_ = nh_.subscribe(config_.lio_status_topic, 10,
                                           &S2pSlamNode::lioStatusCallback, this);
    lidar_subscriber_ = nh_.subscribe(config_.lidar_topic, 2,
                                      &S2pSlamNode::lidarCallback, this);

    depth_subscriber_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(
        nh_, config_.depth_topic, 3);
    rgb_subscriber_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(
        nh_, config_.rgb_topic, 3);
    synchronizer_ = std::make_unique<ImageSynchronizer>(ImageSyncPolicy(10),
                                                        *depth_subscriber_,
                                                        *rgb_subscriber_);
    synchronizer_->registerCallback(boost::bind(
        &S2pSlamNode::imageCallback, this, boost::placeholders::_1,
        boost::placeholders::_2));

    corrected_odom_publisher_ = nh_private_.advertise<nav_msgs::Odometry>(
        "semantic_corrected_odom", 10);
    corrected_path_publisher_ = nh_private_.advertise<nav_msgs::Path>(
        "semantic_corrected_path", 2, true);
    structural_mask_publisher_ = nh_private_.advertise<sensor_msgs::Image>(
        "structural_mask", 2);
    path_.header.frame_id = config_.world_frame;
    ROS_INFO("S2P-SLAM reference node initialized: posterior-only STSM, positive period state, and gated factors enabled");
  }

 private:
  using ImageSyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                       sensor_msgs::Image>;
  using ImageSynchronizer = message_filters::Synchronizer<ImageSyncPolicy>;

  static Config readConfig(ros::NodeHandle& nh) {
    Config config;
    config.load(nh);
    return config;
  }

  void wheelCallback(const nav_msgs::Odometry::ConstPtr& message) {
    std::lock_guard<std::mutex> lock(wheel_mutex_);
    wheel_.stamp = message->header.stamp;
    wheel_.velocity = message->twist.twist.linear.x;
    wheel_.yaw_rate = message->twist.twist.angular.z;
    wheel_.commanded_velocity = latest_command_velocity_;
    wheel_.commanded_yaw_rate = latest_command_yaw_rate_;
    wheel_.valid = std::isfinite(wheel_.velocity) && std::isfinite(wheel_.yaw_rate);
  }

  void commandCallback(const geometry_msgs::Twist::ConstPtr& message) {
    std::lock_guard<std::mutex> lock(wheel_mutex_);
    latest_command_velocity_ = message->linear.x;
    latest_command_yaw_rate_ = message->angular.z;
    wheel_.commanded_velocity = latest_command_velocity_;
    wheel_.commanded_yaw_rate = latest_command_yaw_rate_;
  }

  void lioStatusCallback(const std_msgs::Bool::ConstPtr& message) {
    lio_status_converged_.store(message->data);
  }

  WheelObservation wheelSnapshot(const ros::Time& stamp) const {
    std::lock_guard<std::mutex> lock(wheel_mutex_);
    WheelObservation observation = wheel_;
    if (!observation.valid || observation.stamp.isZero() ||
        std::abs((stamp - observation.stamp).toSec()) > 0.20) {
      observation.velocity = 0.0;
      observation.yaw_rate = 0.0;
      observation.valid = false;
    }
    return observation;
  }

  void lioCallback(const nav_msgs::Odometry::ConstPtr& message) {
    if (!validPoseMessage(message->pose.pose)) {
      ROS_WARN_THROTTLE(1.0, "S2P rejected an invalid LIO pose message");
      return;
    }
    const Pose3 lio_pose = poseFromMsg(message->pose.pose);
    lio_pose_buffer_.add(message->header.stamp, lio_pose);
    const FeatureBundle features = frontend_.bundleAt(message->header.stamp);
    const WheelObservation wheel = wheelSnapshot(message->header.stamp);
    const auto posterior = backend_.process(*message, wheel, features,
                                            lio_status_converged_.load());
    if (!posterior) return;
    posterior_buffer_.add(*posterior);
    publishPosterior(*posterior);
    logger_.log(*posterior, backend_.mode());
  }

  void imageCallback(const sensor_msgs::ImageConstPtr& depth_message,
                     const sensor_msgs::ImageConstPtr& rgb_message) {
    const auto posterior = posterior_buffer_.nearest(depth_message->header.stamp,
                                                     0.20);
    if (!posterior) {
      ROS_WARN_THROTTLE(2.0, "STSM skipped an RGB-D packet without a posterior pose");
      return;
    }
    cv_bridge::CvImageConstPtr rgb_bridge;
    cv_bridge::CvImageConstPtr depth_bridge;
    try {
      rgb_bridge = cv_bridge::toCvShare(rgb_message,
                                        sensor_msgs::image_encodings::BGR8);
      depth_bridge = cv_bridge::toCvShare(depth_message);
    } catch (const cv_bridge::Exception& exception) {
      ROS_WARN_THROTTLE(1.0, "RGB-D conversion failed: %s", exception.what());
      return;
    }
    Pose3 motion_pose = posterior->pose;
    bool motion_pose_is_lio = false;
    if (backend_.mode() == FactorGraphBackend::Mode::kLio) {
      const auto synchronized_lio = lio_pose_buffer_.nearest(
          depth_message->header.stamp, 0.20);
      if (synchronized_lio) {
        motion_pose = *synchronized_lio;
        motion_pose_is_lio = true;
      }
    }
    const auto result = frontend_.processImages(
        rgb_bridge->image, depth_bridge->image, depth_message->header.stamp,
        *posterior, motion_pose, motion_pose_is_lio);
    if (result.structural_mask.empty()) {
      ROS_WARN_THROTTLE(1.0, "RGB and aligned depth dimensions do not match");
      return;
    }
    structural_mask_publisher_.publish(
        cv_bridge::CvImage(depth_message->header,
                           sensor_msgs::image_encodings::MONO8,
                           result.structural_mask)
            .toImageMsg());
    mapper_.processPacket(depth_bridge->image, result.structural_mask,
                          depth_message->header.stamp, *posterior,
                          ++packet_counter_);
  }

  void lidarCallback(const sensor_msgs::PointCloud2::ConstPtr& message) {
    const auto posterior = posterior_buffer_.nearest(message->header.stamp, 0.20);
    const auto lio_pose = lio_pose_buffer_.nearest(message->header.stamp, 0.20);
    if (!posterior || !lio_pose) return;
    pcl::PointCloud<pcl::PointXYZI> input;
    pcl::fromROSMsg(*message, input);
    pcl::PointCloud<pcl::PointXYZI>::Ptr body_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    body_cloud->reserve(input.size());
    const bool already_body = message->header.frame_id == config_.body_frame;
    for (const auto& point : input.points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z)) {
        continue;
      }
      const Point3 input_point(point.x, point.y, point.z);
      const Point3 point_b = already_body ? input_point
                                          : lio_pose->transformTo(input_point);
      pcl::PointXYZI output;
      output.x = static_cast<float>(point_b.x());
      output.y = static_cast<float>(point_b.y());
      output.z = static_cast<float>(point_b.z());
      output.intensity = point.intensity;
      body_cloud->push_back(output);
    }
    frontend_.processLidar(body_cloud, message->header.stamp, *posterior,
                           backend_.activeAisle());
  }

  void publishPosterior(const PosteriorPose& posterior) {
    nav_msgs::Odometry odometry;
    odometry.header.frame_id = config_.world_frame;
    odometry.header.stamp = posterior.stamp;
    odometry.child_frame_id = config_.body_frame;
    odometry.pose.pose = poseToMsg(posterior.pose);
    const Vector3 velocity_body =
        posterior.pose.rotation().unrotate(posterior.velocity);
    odometry.twist.twist.linear.x = velocity_body.x();
    odometry.twist.twist.linear.y = velocity_body.y();
    odometry.twist.twist.linear.z = velocity_body.z();
    // GTSAM Pose3 uses [rotation, translation], whereas ROS covariance uses
    // [translation, rotation].
    constexpr std::array<int, 6> kRosToGtsam{3, 4, 5, 0, 1, 2};
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 6; ++column) {
        odometry.pose.covariance[row * 6 + column] =
            posterior.covariance(kRosToGtsam[row], kRosToGtsam[column]);
      }
    }
    corrected_odom_publisher_.publish(odometry);

    geometry_msgs::PoseStamped pose;
    pose.header = odometry.header;
    pose.pose = odometry.pose.pose;
    path_.header.stamp = posterior.stamp;
    path_.poses.push_back(pose);
    corrected_path_publisher_.publish(path_);

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(posterior.pose.x(), posterior.pose.y(),
                                    posterior.pose.z()));
    const auto quaternion = posterior.pose.rotation().toQuaternion();
    transform.setRotation(tf::Quaternion(quaternion.x(), quaternion.y(),
                                         quaternion.z(), quaternion.w()));
    broadcaster_.sendTransform(tf::StampedTransform(
        transform, posterior.stamp, config_.world_frame, config_.body_frame));
  }

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  Config config_;
  FactorGraphBackend backend_;
  StructureFrontend frontend_;
  SpatiotemporalSemanticMapper mapper_;
  RunLogger logger_;
  ros::Subscriber lio_subscriber_;
  ros::Subscriber wheel_subscriber_;
  ros::Subscriber command_subscriber_;
  ros::Subscriber lio_status_subscriber_;
  ros::Subscriber lidar_subscriber_;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>>
      depth_subscriber_;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>>
      rgb_subscriber_;
  std::unique_ptr<ImageSynchronizer> synchronizer_;
  ros::Publisher corrected_odom_publisher_;
  ros::Publisher corrected_path_publisher_;
  ros::Publisher structural_mask_publisher_;
  tf::TransformBroadcaster broadcaster_;
  mutable std::mutex wheel_mutex_;
  WheelObservation wheel_;
  double latest_command_velocity_ = 0.0;
  double latest_command_yaw_rate_ = 0.0;
  std::atomic<bool> lio_status_converged_{true};
  std::atomic<std::uint64_t> packet_counter_{0};
  TimedPoseBuffer lio_pose_buffer_;
  PosteriorBuffer posterior_buffer_;
  nav_msgs::Path path_;
};

}  // namespace s2p

int main(int argc, char** argv) {
  ros::init(argc, argv, "semantic_octomap_node");
  ros::NodeHandle nh_private("~");
  s2p::S2pSlamNode node(nh_private);
  ros::AsyncSpinner spinner(4);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
