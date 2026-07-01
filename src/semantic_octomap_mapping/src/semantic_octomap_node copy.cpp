#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h> 
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>               
#include <nav_msgs/Odometry.h>           
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h> 
#include <visualization_msgs/MarkerArray.h>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h> 
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>    
#include <deque>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <mutex>

// GTSAM Headers (3D + IMU)
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>

// Message Filters
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace gtsam;

// =======================================================================================
// 全局配置中心 (v36.0 - Final Robust VINS + RGB Ground Check)
// =======================================================================================
struct SystemConfig {
    double leg_spacing;
    double lidar_z_min; 
    double lidar_z_max; 
    double z_ceiling;        
    double z_obstacle_thresh;
    double ground_offset_k;
    double ground_safe_height_ratio;
    double ground_safe_width_ratio;
    double camera_z_shift; 
    double hole_fill_area;
    int hsv_s_min;
    int hsv_v_min;
    int kernel_far_w, kernel_far_h;
    int kernel_near_w, kernel_near_h;
    double roi_h_ratio;
    double roi_w_ratio;
    double trigger_thresh;
    double cooldown_time;
    double motion_deadzone;
    double noise_vision_std;    
    double outlier_thresh; 
    double semantic_hit_prob;   
    double semantic_miss_prob;  
    double semantic_thresh;     
    
    // IMU 参数
    double imu_acc_noise;
    double imu_gyro_noise;
    double imu_acc_bias_noise;
    double imu_gyro_bias_noise;
    double imu_dt; 
    
    double dust_intensity_thresh;
    double prob_miss_static;   
    double prob_miss_dynamic;  
    float log_odds_miss_static;
    float log_odds_miss_dynamic;
    float log_odds_hit;         
    float log_odds_min;         
    float log_odds_max;
    std::string global_frame_id;
    double vp_weight_base;      
};

SystemConfig config;

void loadParameters(ros::NodeHandle& nh) {
    nh.param<double>("leg_spacing", config.leg_spacing, 1.66);
    nh.param<double>("lidar_z_min", config.lidar_z_min, 0.2); 
    nh.param<double>("lidar_z_max", config.lidar_z_max, 2.5); 
    nh.param<double>("z_ceiling", config.z_ceiling, 3.0); 
    nh.param<double>("z_obstacle_thresh", config.z_obstacle_thresh, 2.05);
    nh.param<double>("ground_offset_k", config.ground_offset_k, 0.35);
    nh.param<double>("ground_safe_height_ratio", config.ground_safe_height_ratio, 0.85); 
    nh.param<double>("ground_safe_width_ratio", config.ground_safe_width_ratio, 0.50); 
    nh.param<double>("camera_z_shift", config.camera_z_shift, 0.0); 
    nh.param<double>("hole_fill_area", config.hole_fill_area, 4000.0);
    nh.param<int>("hsv_s_min", config.hsv_s_min, 40);
    nh.param<int>("hsv_v_min", config.hsv_v_min, 30);
    nh.param<int>("kernel_far_w", config.kernel_far_w, 1);
    nh.param<int>("kernel_far_h", config.kernel_far_h, 3);
    nh.param<int>("kernel_near_w", config.kernel_near_w, 2);
    nh.param<int>("kernel_near_h", config.kernel_near_h, 10);
    nh.param<double>("roi_h_ratio", config.roi_h_ratio, 0.12);
    nh.param<double>("roi_w_ratio", config.roi_w_ratio, 0.08);
    nh.param<double>("trigger_thresh", config.trigger_thresh, 0.10);
    nh.param<double>("cooldown_time", config.cooldown_time, 0.5);
    nh.param<double>("motion_deadzone", config.motion_deadzone, 0.05);
    nh.param<double>("noise_vision_std", config.noise_vision_std, 0.1); 
    nh.param<double>("outlier_thresh", config.outlier_thresh, 0.8);     
    nh.param<double>("semantic_hit_prob", config.semantic_hit_prob, 0.2); 
    nh.param<double>("semantic_miss_prob", config.semantic_miss_prob, 0.05); 
    nh.param<double>("semantic_thresh", config.semantic_thresh, 0.75); 
    nh.param<double>("dust_intensity_thresh", config.dust_intensity_thresh, 0.5);
    nh.param<double>("prob_miss_static", config.prob_miss_static, 0.46); 
    nh.param<double>("prob_miss_dynamic", config.prob_miss_dynamic, 0.35); 
    nh.param<double>("vp_weight_base", config.vp_weight_base, 0.1); 
    
    // IMU 默认参数
    nh.param<double>("imu_acc_noise", config.imu_acc_noise, 0.01);
    nh.param<double>("imu_gyro_noise", config.imu_gyro_noise, 0.001);
    nh.param<double>("imu_acc_bias_noise", config.imu_acc_bias_noise, 1.0e-4);
    nh.param<double>("imu_gyro_bias_noise", config.imu_gyro_bias_noise, 1.0e-5);
    nh.param<double>("imu_dt", config.imu_dt, 0.01); 

    // [重要] 默认使用 camera_init，适配 LOAM/LIO-SAM
    nh.param<std::string>("global_frame_id", config.global_frame_id, "camera_init"); 

    auto probToLogOdds = [](double p) { return std::log(p / (1.0 - p)); };
    config.log_odds_miss_static = probToLogOdds(config.prob_miss_static);
    config.log_odds_miss_dynamic = probToLogOdds(config.prob_miss_dynamic);
    config.log_odds_hit = probToLogOdds(0.7);      
    config.log_odds_min = probToLogOdds(0.12);      
    config.log_odds_max = probToLogOdds(0.97);      

    ROS_INFO("Config v36.0 (Robust VINS + RGB Check) Loaded.");
}

// =======================================================================================
// 模块: VanishingPointFactor (3D)
// =======================================================================================
class VanishingPointFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    double measured_yaw_; 
public:
    VanishingPointFactor(gtsam::Key poseKey, double measured_yaw, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey), measured_yaw_(measured_yaw) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        Rot3 R = pose.rotation();
        double predicted_yaw = R.yaw();
        double error = predicted_yaw - measured_yaw_;
        while (error > M_PI) error -= 2 * M_PI;
        while (error < -M_PI) error += 2 * M_PI;

        if (H) {
            *H = (gtsam::Matrix(1, 6) << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0).finished();
        }
        return gtsam::Vector1(error);
    }
};

// =======================================================================================
// 模块: PlanarConstraintFactor (地面约束)
// =======================================================================================
class PlanarConstraintFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    double z_target_;
public:
    PlanarConstraintFactor(gtsam::Key key, double z_target, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), z_target_(z_target) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        double z_err = pose.z() - z_target_;
        double roll_err = pose.rotation().roll();
        double pitch_err = pose.rotation().pitch();
        
        if (H) {
             *H = (gtsam::Matrix(3, 6) << 
                 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Roll
                 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // Pitch
                 0.0, 0.0, 0.0, 0.0, 0.0, 1.0  // Z
             ).finished();
        }
        return (gtsam::Vector(3) << roll_err, pitch_err, z_err).finished();
    }
};

// =======================================================================================
// 模块: VanishingPointDetector
// =======================================================================================
class VanishingPointDetector {
private:
    double smoothed_yaw_;       
    bool is_initialized_;
    const double EMA_ALPHA = 0.05; 

public:
    VanishingPointDetector() : smoothed_yaw_(0.0), is_initialized_(false) {}

    struct VPResult {
        bool valid;
        double yaw; double raw_yaw; cv::Point2f pt; int inliers;    
    };

    VPResult detect(cv::Mat& img_for_debug) { 
        VPResult res = {false, 0.0, 0.0, cv::Point2f(0,0), 0};
        cv::Mat gray;
        if (img_for_debug.channels() == 3) cv::cvtColor(img_for_debug, gray, cv::COLOR_BGR2GRAY);
        else gray = img_for_debug.clone();
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, gray);
        cv::Mat edges; cv::Canny(gray, edges, 60, 180, 3);
        
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180, 40, 60, 30);
        if (lines.empty()) return res;

        struct LineObj { cv::Vec4f l; float len; float weight; };
        std::vector<LineObj> candidates;
        for (const auto& l_raw : lines) {
            cv::Vec4f l(l_raw[0], l_raw[1], l_raw[2], l_raw[3]);
            float dx = l[2]-l[0]; float dy = l[3]-l[1];
            float len = std::sqrt(dx*dx + dy*dy);
            float angle_deg = std::atan2(dy, dx) * 180.0 / CV_PI;
            if (std::abs(angle_deg) < 20 || std::abs(angle_deg) > 75) continue;
            float weight = std::pow(len, 1.5); 
            candidates.push_back({l, len, weight});
        }
        if (candidates.size() < 4) return res;

        int w = gray.cols; int h = gray.rows;
        double best_total_weight = 0.0; int best_inlier_count = 0; cv::Point2f best_vp(w/2, h/2);
        int iter_num = std::min(300, (int)candidates.size() * 10);
        
        for (int i = 0; i < iter_num; ++i) { 
            int idx1 = rand() % candidates.size(); int idx2 = rand() % candidates.size();
            if (idx1 == idx2) continue;
            cv::Point2f vp = getIntersection(candidates[idx1].l, candidates[idx2].l);
            if (vp.y < h * 0.30 || vp.y > h * 0.70) continue;
            if (vp.x < -w || vp.x > 2*w) continue;
            double current_weight = 0.0; int current_count = 0;
            for (const auto& obj : candidates) {
                if (distPointToLine(vp, obj.l) < 10.0) { current_weight += obj.weight; current_count++; }
            }
            if (current_weight > best_total_weight) { best_total_weight = current_weight; best_inlier_count = current_count; best_vp = vp; }
        }

        if (best_inlier_count > 5) { 
            res.valid = true; res.pt = best_vp; res.inliers = best_inlier_count;
            double fx = 904.3275; double cx = 653.1630;
            double measured_yaw = -std::atan((best_vp.x - cx) / fx);
            res.raw_yaw = measured_yaw;
            if (!is_initialized_) { smoothed_yaw_ = measured_yaw; is_initialized_ = true; } 
            else {
                double diff = std::abs(measured_yaw - smoothed_yaw_);
                double dynamic_alpha = (diff > 0.44) ? 0.05 : EMA_ALPHA;
                smoothed_yaw_ = dynamic_alpha * measured_yaw + (1.0 - dynamic_alpha) * smoothed_yaw_;
            }
            res.yaw = smoothed_yaw_;
            for (const auto& obj : candidates) {
                if (distPointToLine(best_vp, obj.l) < 10.0) cv::line(img_for_debug, cv::Point(obj.l[0], obj.l[1]), cv::Point(obj.l[2], obj.l[3]), cv::Scalar(255, 0, 255), 2);
            }
        }
        return res;
    }
    void reset() { is_initialized_ = false; }
private:
    cv::Point2f getIntersection(cv::Vec4f l1, cv::Vec4f l2) {
        float x1=l1[0], y1=l1[1], x2=l1[2], y2=l1[3]; float x3=l2[0], y3=l2[1], x4=l2[2], y4=l2[3];
        float d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
        if (std::abs(d) < 1e-5) return cv::Point2f(0, 0); 
        return cv::Point2f(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/d, ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/d);
    }
    float distPointToLine(cv::Point2f p, cv::Vec4f l) {
        float A = p.x - l[0]; float B = p.y - l[1]; float C = l[2] - l[0]; float D = l[3] - l[1];
        float len_sq = C * C + D * D; if (len_sq == 0) return -1;
        return std::abs(C * (l[1] - p.y) - (l[0] - p.x) * D) / std::sqrt(len_sq);
    }
};

// =======================================================================================
// [ROBUST v4] 模块: FactorGraphTracker (Stabilized VINS)
// =======================================================================================
class FactorGraphTracker {
public:
    FactorGraphTracker() : key_index_(0), last_keyframe_time_(0) {
        initParams();
        // 初始化先验
        hardReset(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0,0,0)), 
                  gtsam::Vector3(0,0,0), 
                  gtsam::imuBias::ConstantBias());
    }

    void addImuMeasurement(const sensor_msgs::Imu::ConstPtr& msg) {
        Vector3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        Vector3 gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (preintegrator_) {
            preintegrator_->integrateMeasurement(acc, gyro, config.imu_dt);
        }
    }

    void predictAndUpdate(ros::Time now, double semantic_correction_x, bool has_vp, double vp_yaw, double vp_conf) {
        if (last_keyframe_time_.isZero()) {
            last_keyframe_time_ = now;
            return;
        }

        double dt_since_keyframe = (now - last_keyframe_time_).toSec();

        // [改进 1] 增大关键帧间隔到 0.3s (增加积分长度，提高稳定性)
        if (dt_since_keyframe < 0.3) {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            NavState prev_state(current_pose_, current_velocity_);
            NavState prop_state = preintegrator_->predict(prev_state, current_bias_);
            latest_display_pose_ = prop_state.pose();
            return; 
        }

        key_index_++;
        last_keyframe_time_ = now;

        gtsam::NonlinearFactorGraph new_factors;
        gtsam::Values new_values;

        // 1. IMU 因子 & 状态预测
        NavState prev_state(current_pose_, current_velocity_);
        NavState prop_state; // 预测状态
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            CombinedImuFactor imu_factor(X(key_index_-1), V(key_index_-1), 
                                         X(key_index_),   V(key_index_), 
                                         B(key_index_-1), B(key_index_), 
                                         *preintegrator_);
            new_factors.add(imu_factor);

            prop_state = preintegrator_->predict(prev_state, current_bias_);
            
            // [改进 2] 添加 "Soft Odometry Factor" (相对位姿约束)
            // 防止 IMU 积分发散/奇异
            Pose3 relative_pose = current_pose_.between(prop_state.pose());
            auto soft_odom_noise = noiseModel::Diagonal::Sigmas(
                (Vector(6) << 0.1, 0.1, 0.1, 0.2, 0.2, 0.2).finished());
            
            new_factors.add(BetweenFactor<Pose3>(X(key_index_-1), X(key_index_), relative_pose, soft_odom_noise));

            // 重置积分器
            preintegrator_->resetIntegrationAndSetBias(current_bias_);
        }

        // 更新当前状态
        current_pose_ = prop_state.pose();
        current_velocity_ = prop_state.velocity();

        new_values.insert(X(key_index_), current_pose_);
        new_values.insert(V(key_index_), current_velocity_);
        new_values.insert(B(key_index_), current_bias_);

        // 2. 视觉 VP 约束
        if (has_vp) {
            double sigma = config.vp_weight_base / (vp_conf + 1e-3);
            auto vp_noise = noiseModel::Diagonal::Sigmas(Vector1(sigma));
            new_factors.add(boost::make_shared<VanishingPointFactor>(X(key_index_), vp_yaw, vp_noise));
        }

        // 3. 笼腿位置约束
        if (std::abs(semantic_correction_x) > 0.001) {
            new_factors.add(boost::make_shared<PlanarConstraintFactor>(X(key_index_), 0.0, 
                noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05)))); 
        }
        
        // 4. 地面约束 (Z, Roll, Pitch)
        new_factors.add(boost::make_shared<PlanarConstraintFactor>(X(key_index_), 0.0, 
                noiseModel::Diagonal::Sigmas(Vector3(0.05, 0.05, 0.01)))); 

        // 5. 执行优化 (带保护)
        try {
            isam_->update(new_factors, new_values);
            isam_->update(); 
            
            current_pose_ = isam_->calculateEstimate<Pose3>(X(key_index_));
            current_velocity_ = isam_->calculateEstimate<Vector3>(V(key_index_));
            current_bias_ = isam_->calculateEstimate<imuBias::ConstantBias>(B(key_index_));
            
            latest_display_pose_ = current_pose_;

        } catch (gtsam::IndeterminantLinearSystemException& e) {
            ROS_WARN_THROTTLE(5.0, "GTSAM Singularity. Hard Resetting...");
            hardReset(current_pose_, current_velocity_, current_bias_);
        } catch (std::exception& e) {
            ROS_ERROR_THROTTLE(5.0, "GTSAM Failure: %s", e.what());
            hardReset(current_pose_, current_velocity_, current_bias_);
        }
    }

    gtsam::Pose3 getPose() { return latest_display_pose_; }

private:
    static Symbol X(int i) { return Symbol('x', i); }
    static Symbol V(int i) { return Symbol('v', i); }
    static Symbol B(int i) { return Symbol('b', i); }

    void initParams() {
        auto p = gtsam::PreintegrationCombinedParams::MakeSharedU(9.81); 
        p->accelerometerCovariance = I_3x3 * pow(config.imu_acc_noise, 2);
        p->gyroscopeCovariance = I_3x3 * pow(config.imu_gyro_noise, 2);
        p->integrationCovariance = I_3x3 * 1e-8;
        p->biasAccCovariance = I_3x3 * pow(config.imu_acc_bias_noise, 2);
        p->biasOmegaCovariance = I_3x3 * pow(config.imu_gyro_bias_noise, 2);
        p->biasAccOmegaInt = I_6x6 * 1e-5;
        imu_params_ = p;
    }

    void hardReset(const Pose3& pose, const Vector3& vel, const imuBias::ConstantBias& bias) {
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.1;
        params.relinearizeSkip = 1;
        isam_ = std::make_unique<gtsam::ISAM2>(params);

        current_pose_ = pose;
        current_velocity_ = vel;
        current_bias_ = bias;
        latest_display_pose_ = pose;

        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            preintegrator_ = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(imu_params_, current_bias_);
        }

        gtsam::NonlinearFactorGraph new_graph;
        gtsam::Values new_values;

        auto pose_noise = noiseModel::Diagonal::Sigmas((Vector(6) << 0.001, 0.001, 0.001, 0.01, 0.01, 0.01).finished());
        auto vel_noise  = noiseModel::Isotropic::Sigma(3, 0.05);
        auto bias_noise = noiseModel::Isotropic::Sigma(6, 1e-3); 

        new_graph.add(PriorFactor<Pose3>(X(key_index_), current_pose_, pose_noise));
        new_graph.add(PriorFactor<Vector3>(V(key_index_), current_velocity_, vel_noise));
        new_graph.add(PriorFactor<imuBias::ConstantBias>(B(key_index_), current_bias_, bias_noise));

        new_values.insert(X(key_index_), current_pose_);
        new_values.insert(V(key_index_), current_velocity_);
        new_values.insert(B(key_index_), current_bias_);

        isam_->update(new_graph, new_values);
        new_graph.resize(0); new_values.clear();
        
        ROS_INFO("GTSAM System Initialized/Reset at Key: %lu", key_index_);
    }

    std::unique_ptr<gtsam::ISAM2> isam_;
    boost::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params_;
    std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> preintegrator_;
    
    unsigned long key_index_;
    gtsam::Pose3 current_pose_;
    gtsam::Pose3 latest_display_pose_; 
    gtsam::Vector3 current_velocity_;
    gtsam::imuBias::ConstantBias current_bias_;
    
    ros::Time last_keyframe_time_; 
    std::mutex imu_mutex_;
};

// =======================================================================================
// 模块: StructureOdometer
// =======================================================================================
class StructureOdometer {
public:
    StructureOdometer(ros::NodeHandle& nh) 
        : nh_(nh), is_active_(false), last_fix_amount_(0.0) {
        
        pub_corrected_path_ = nh_.advertise<nav_msgs::Path>("semantic_corrected_path", 1);
        pub_corrected_odom_ = nh_.advertise<nav_msgs::Odometry>("semantic_corrected_odom", 1);
        
        sub_imu_ = nh_.subscribe("/imu", 100, &StructureOdometer::imuCallback, this);
        
        if (config.global_frame_id.empty()) config.global_frame_id = "camera_init";
        corrected_path_.header.frame_id = config.global_frame_id;
        
        last_proc_time_ = ros::Time::now();
        last_trigger_time_ = ros::Time(0);
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        tracker_.addImuMeasurement(msg);
    }

    void process(cv::Mat& rgb_img, cv::Mat& obstacle_img, const tf::Transform& tf_cam) {
        ros::Time now = ros::Time::now();

        // 1. 结构物检测
        int h = obstacle_img.rows; int w = obstacle_img.cols;
        int roi_h = h * config.roi_h_ratio; int roi_w = w * config.roi_w_ratio; int roi_y = h - roi_h - 5;   
        cv::Rect roi_L(5, roi_y, roi_w, roi_h); cv::Rect roi_R(w - roi_w - 5, roi_y, roi_w, roi_h);
        roi_L &= cv::Rect(0, 0, w, h); roi_R &= cv::Rect(0, 0, w, h);
        float ratio_L = calculateOccupancyRatio(obstacle_img(roi_L));
        float ratio_R = calculateOccupancyRatio(obstacle_img(roi_R));
        double measurement = std::max(ratio_L, ratio_R); 

        // 2. 消失点检测
        VanishingPointDetector::VPResult vp_res = vp_detector_.detect(rgb_img);
        
        // 3. 计算语义修正量
        bool leg_detected = (measurement > config.trigger_thresh);
        double correction_signal = 0.0;
        double time_since_last = (now - last_trigger_time_).toSec();
        
        if (!is_active_ && leg_detected) {
            is_active_ = true; 
        } else if (is_active_ && !leg_detected) {
            if (time_since_last > config.cooldown_time) { 
                double cur_x = tracker_.getPose().x();
                int idx = std::round(cur_x / config.leg_spacing);
                double anchor = idx * config.leg_spacing;
                if (std::abs(cur_x - anchor) < config.outlier_thresh) {
                    correction_signal = 1.0; 
                    last_fix_amount_ = anchor - cur_x; 
                }
                last_trigger_time_ = now;
                is_active_ = false;
            } else {
                is_active_ = false;
            }
        }

        // 4. 图优化更新
        double vp_conf = vp_res.valid ? std::min(1.0, vp_res.inliers / 30.0) : 0.0;
        tracker_.predictAndUpdate(now, correction_signal, vp_res.valid, vp_res.yaw, vp_conf);

        // 5. 获取结果
        gtsam::Pose3 opt_pose = tracker_.getPose();
        gtsam::Quaternion q = opt_pose.rotation().toQuaternion();
        
        publishCorrectedData(opt_pose.x(), opt_pose.y(), opt_pose.z(), q);
        drawVisualization(rgb_img, roi_L, roi_R, measurement, opt_pose.x(), last_fix_amount_, vp_res);
    }

private:
    float calculateOccupancyRatio(const cv::Mat& roi) {
        int red = 0;
        for(int i=0; i<roi.rows; ++i) {
            for(int j=0; j<roi.cols; ++j) {
                cv::Vec3b p = roi.at<cv::Vec3b>(i,j);
                if(p[2] > 200 && p[1] < 50) red++; 
            }
        }
        return (float)red / (roi.rows * roi.cols);
    }
    
    void drawVisualization(cv::Mat& img, cv::Rect rL, cv::Rect rR, double meas, double cor_dist, double fix, 
                           const VanishingPointDetector::VPResult& vp_res) {
        cv::Scalar col = (meas > config.trigger_thresh) ? cv::Scalar(0,0,255) : cv::Scalar(255,0,0);
        int th = (meas > config.trigger_thresh) ? 3 : 2;
        cv::rectangle(img, rL, col, th); cv::rectangle(img, rR, col, th);
        if (vp_res.valid) {
            cv::line(img, cv::Point(img.cols/2, img.rows), vp_res.pt, cv::Scalar(255, 0, 255), 2); 
            cv::circle(img, vp_res.pt, 5, cv::Scalar(0, 255, 255), -1);
            cv::circle(img, vp_res.pt, 12, cv::Scalar(0, 0, 255), 2);
        }
        std::string s_vp = vp_res.valid ? "VP Yaw: " + std::to_string(vp_res.yaw * 57.3) + " deg" : "VP: Searching...";
        std::string s_dist = "Map X: " + std::to_string(cor_dist) + "m";
        std::string s_fix = "Leg Fix: " + std::to_string(fix);
        
        int font = cv::FONT_HERSHEY_SIMPLEX; double font_scale = 0.8; int thickness = 2; int baseline = 0;
        auto drawCenteredText = [&](std::string text, int y_offset, cv::Scalar color) {
            cv::Size textSize = cv::getTextSize(text, font, font_scale, thickness, &baseline);
            int text_x = (img.cols - textSize.width) / 2;
            int text_y = (img.rows / 2) + y_offset;
            cv::putText(img, text, cv::Point(text_x, text_y), font, font_scale, cv::Scalar(0,0,0), thickness + 2);
            cv::putText(img, text, cv::Point(text_x, text_y), font, font_scale, color, thickness);
        };
        drawCenteredText("GTSAM VINS-Fusion", -60, cv::Scalar(0, 255, 0));
        drawCenteredText(s_vp, -20, cv::Scalar(0, 255, 255));
        drawCenteredText(s_dist, 20, cv::Scalar(0, 0, 255));
        drawCenteredText(s_fix, 60, cv::Scalar(100, 0, 100));
    }

    void publishCorrectedData(float x, float y, float z, gtsam::Quaternion q) {
        nav_msgs::Odometry odom;
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = config.global_frame_id;
        odom.child_frame_id = "body_corrected";
        odom.pose.pose.position.x = x; odom.pose.pose.position.y = y; odom.pose.pose.position.z = z;
        odom.pose.pose.orientation.x = q.x(); odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z(); odom.pose.pose.orientation.w = q.w();
        pub_corrected_odom_.publish(odom);
        
        geometry_msgs::PoseStamped pose;
        pose.header = odom.header; pose.pose = odom.pose.pose;
        corrected_path_.header.stamp = ros::Time::now();
        corrected_path_.poses.push_back(pose);
        pub_corrected_path_.publish(corrected_path_);
    }

    ros::NodeHandle nh_;
    ros::Publisher pub_corrected_path_, pub_corrected_odom_;
    ros::Subscriber sub_imu_; 
    nav_msgs::Path corrected_path_;
    FactorGraphTracker tracker_;
    VanishingPointDetector vp_detector_;
    ros::Time last_proc_time_;
    bool is_active_;
    ros::Time last_trigger_time_;
    double last_fix_amount_;
};

// =======================================================================================
// 模块: SemanticOctomapServer
// =======================================================================================
class SemanticOctomapServer {
public:
    SemanticOctomapServer() : smoothed_a_(0.0), smoothed_b_(0.0), smoothed_c_(0.0), semantic_odom_(nh_) {
        ros::NodeHandle nh_private("~"); 
        loadParameters(nh_private);
        nh_ = ros::NodeHandle();
        
        octree_ = new octomap::ColorOcTree(0.05);
        octree_->setProbHit(0.7); octree_->setProbMiss(0.4); 
        octree_->setClampingThresMin(config.log_odds_min); octree_->setClampingThresMax(config.log_odds_max);

        sub_lidar_ = nh_.subscribe("/cloud_registered_effect_world", 1, &SemanticOctomapServer::lidarCallback, this);

        sub_depth_filter_ = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "/camera/aligned_depth_to_color/image_raw", 1);
        sub_rgb_filter_   = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "/camera/color/image_raw", 1);
        sync_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *sub_depth_filter_, *sub_rgb_filter_);
        sync_->registerCallback(boost::bind(&SemanticOctomapServer::syncCallback, this, _1, _2));

        pub_vis_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("octomap_markers", 1, true);
        pub_octomap_ = nh_.advertise<octomap_msgs::Octomap>("octomap_full", 1, true);
        pub_2d_map_ = nh_.advertise<nav_msgs::OccupancyGrid>("projected_map", 1, true);
        pub_debug_img_ = nh_.advertise<sensor_msgs::Image>("traversability_debug", 1);
        pub_vp_debug_ = nh_.advertise<sensor_msgs::Image>("vp_debug_rgb", 1);
        pub_marker_ = nh_.advertise<visualization_msgs::Marker>("nav_center_line", 1);

        ROS_INFO("Final Semantic Navigator (v36.0 - Robust VINS) Started!");
    }

    ~SemanticOctomapServer() { 
        delete octree_; delete sub_depth_filter_; delete sub_rgb_filter_; delete sync_;
    }

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    // 辅助函数: 计算颜色差异
    float colorDist(const cv::Vec3b& a, const cv::Vec3b& b) {
        float dH = std::abs(a[0] - b[0]);
        if (dH > 90) dH = 180 - dH; 
        return std::sqrt(dH*dH*2.0 + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2])*0.5);
    }

    // [核心改进] 包含 RGB 校验的同步回调
    void syncCallback(const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg) {
        cv_bridge::CvImagePtr cv_depth_ptr, cv_rgb_ptr;
        try { 
            cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1); 
            cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8); 
        } catch (...) { return; }

        tf::StampedTransform transform;
        try {
            tf_listener_.waitForTransform(config.global_frame_id, depth_msg->header.frame_id, depth_msg->header.stamp, ros::Duration(0.1));
            tf_listener_.lookupTransform(config.global_frame_id, depth_msg->header.frame_id, depth_msg->header.stamp, transform);
        } catch (...) { return; }

        cv::Mat depth_img = cv_depth_ptr->image;
        cv::Mat rgb_img = cv_rgb_ptr->image; 

        // 1. RGB 预处理与地面采样
        cv::Mat hsv_img;
        cv::cvtColor(rgb_img, hsv_img, cv::COLOR_BGR2HSV);
        int sample_w = 40; int sample_h = 20;
        int sample_x = (hsv_img.cols - sample_w) / 2;
        int sample_y = hsv_img.rows - sample_h - 10;
        cv::Rect ground_roi(sample_x, sample_y, sample_w, sample_h);
        cv::Scalar ground_mean_scalar = cv::mean(hsv_img(ground_roi));
        cv::Vec3b ground_color(ground_mean_scalar[0], ground_mean_scalar[1], ground_mean_scalar[2]);

        cv::Mat debug_depth_viz = cv::Mat::zeros(depth_img.size(), CV_8UC3);
        float fx = 904.3275; float fy = 903.9418; float cx = 653.1630; float cy = 369.6624; int step = 5; 
        std::vector<cv::Point2f> obstacle_pts;
        float color_similarity_thresh = 40.0; 

        for (int v = step; v < depth_img.rows - step; v += step) {
            for (int u = step; u < depth_img.cols - step; u += step) {
                unsigned short d = depth_img.at<unsigned short>(v, u);
                if (d < 300 || d > 3500) continue; 

                float z_cam = d * 0.001f;
                float x_cam = (u - cx) * z_cam / fx;
                float y_cam = (v - cy) * z_cam / fy;
                tf::Vector3 p_c(x_cam, y_cam, z_cam);
                tf::Vector3 p_w = transform * p_c;
                
                float v_norm = (float)v / depth_img.rows;
                float ground_offset = config.ground_offset_k * pow(v_norm, 2); 
                float z_academic = p_w.z() + (z_cam * 0.18) + ground_offset;

                bool is_obstacle = false;
                if (z_academic < config.z_obstacle_thresh) is_obstacle = true;

                // [RGB + 几何 联合校验]
                if (is_obstacle && d < 1500) {
                    cv::Vec3b pixel_color = hsv_img.at<cv::Vec3b>(v, u);
                    float diff = colorDist(pixel_color, ground_color);
                    if (p_w.z() < 0.15 && diff < color_similarity_thresh) {
                        is_obstacle = false; 
                    }
                }

                if (is_obstacle) cv::circle(debug_depth_viz, cv::Point(u, v), 3, cv::Scalar(0, 0, 255), -1);
                else cv::circle(debug_depth_viz, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), -1);
            }
        }

        cv::Mat processed_obstacle_img = enhanceImageAdaptive(debug_depth_viz);
        pub_debug_img_.publish(cv_bridge::CvImage(depth_msg->header, "bgr8", processed_obstacle_img).toImageMsg());
        
        // Update Octomap
        for (int v = step; v < processed_obstacle_img.rows - step; v += step) {
            for (int u = step; u < processed_obstacle_img.cols - step; u += step) {
                cv::Vec3b pixel = processed_obstacle_img.at<cv::Vec3b>(v, u);
                if (pixel[2] > 200 && pixel[1] < 50) { 
                    unsigned short d = depth_img.at<unsigned short>(v, u);
                    if (d == 0 || d > 3500) continue; 
                    float z_cam = d * 0.001f;
                    float x_cam = (u - cx) * z_cam / fx; float y_cam = (v - cy) * z_cam / fy;
                    tf::Vector3 p_c(x_cam, y_cam, z_cam); tf::Vector3 p_w = transform * p_c;
                    if (p_w.z() > config.z_ceiling) continue; 
                    obstacle_pts.push_back(cv::Point2f(p_w.x(), p_w.y()));
                    
                    octomap::point3d endpoint(p_w.x(), p_w.y(), p_w.z() + config.camera_z_shift);
                    octomap::OcTreeKey key;
                    if (octree_->coordToKeyChecked(endpoint, key)) {
                        float& prob = semantic_probs_[key];
                        prob += config.semantic_hit_prob;
                        if (prob > 1.0) prob = 1.0; 
                        octomap::ColorOcTreeNode* n = octree_->updateNode(endpoint, true);
                        if (n) {
                            if (prob > config.semantic_thresh) n->setColor(255, 0, 0); 
                            else if (!n->isColorSet()) n->setColor(0, 255, 255);
                        }
                    }
                }
            }
        }

        if (obstacle_pts.size() > 40) {
            fitNonlinearCenterLine(obstacle_pts, depth_msg->header, processed_obstacle_img, transform);
        }

        semantic_odom_.process(rgb_img, processed_obstacle_img, transform);
        pub_vp_debug_.publish(cv_bridge::CvImage(depth_msg->header, "bgr8", rgb_img).toImageMsg());
    }

    void lidarCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);
        octomap::point3d sensor_origin;
        try {
            tf::StampedTransform tf_sensor;
            tf_listener_.waitForTransform(config.global_frame_id, msg->header.frame_id, msg->header.stamp, ros::Duration(0.05));
            tf_listener_.lookupTransform(config.global_frame_id, msg->header.frame_id, msg->header.stamp, tf_sensor);
            sensor_origin = octomap::point3d(tf_sensor.getOrigin().x(), tf_sensor.getOrigin().y(), tf_sensor.getOrigin().z());
        } catch (tf::TransformException& ex) { return; }

        for (const auto& p : pcl_cloud.points) {
            if (std::abs(p.x) > 50.0 || p.z < config.lidar_z_min || p.z > config.lidar_z_max) continue;
            if (p.intensity < config.dust_intensity_thresh) continue; 
            octomap::point3d endpoint(p.x, p.y, p.z);
            octomap::KeyRay ray;
            if (octree_->computeRayKeys(sensor_origin, endpoint, ray)) {
                for (const auto& key : ray) {
                    octomap::ColorOcTreeNode* node = octree_->search(key);
                    if (node && octree_->isNodeOccupied(node)) {
                        bool is_cage_leg = false;
                        if (semantic_probs_.count(key) && semantic_probs_[key] > config.semantic_thresh) is_cage_leg = true;
                        float log_odds_update = is_cage_leg ? config.log_odds_miss_static : config.log_odds_miss_dynamic;
                        float next_val = node->getLogOdds() + log_odds_update;
                        if (next_val < config.log_odds_min) next_val = config.log_odds_min;
                        if (next_val > config.log_odds_max) next_val = config.log_odds_max;
                        node->setLogOdds(next_val);
                    }
                }
            }
            octree_->updateNode(endpoint, true); 
        }
        for (const auto& p : pcl_cloud.points) { 
            if (p.z < config.lidar_z_min || p.z > config.lidar_z_max || p.intensity < config.dust_intensity_thresh) continue;
            octomap::point3d endpoint(p.x, p.y, p.z);
            octomap::ColorOcTreeNode* n = octree_->search(endpoint);
            if (n) {
                octomap::OcTreeKey key;
                if (octree_->coordToKeyChecked(endpoint, key)) {
                    if (semantic_probs_.count(key) && semantic_probs_[key] > config.semantic_thresh) n->setColor(255, 0, 0); 
                    else n->setColor(0, 255, 255); 
                }
            }
        }
        octree_->updateInnerOccupancy();
        publishMarkers(msg->header);
    }

    cv::Mat enhanceImageAdaptive(const cv::Mat& src) {
        cv::Mat dst = src.clone();
        cv::Mat black_mask;
        cv::inRange(src, cv::Scalar(0, 0, 0), cv::Scalar(10, 10, 10), black_mask);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(black_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& cnt : contours) {
            if (cv::contourArea(cnt) < config.hole_fill_area) {
                cv::drawContours(dst, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(0, 255, 0), -1);
            }
        }
        cv::Mat hsv, red_mask1, red_mask2, red_mask;
        cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, config.hsv_s_min, config.hsv_v_min), cv::Scalar(10, 255, 255), red_mask1);
        cv::inRange(hsv, cv::Scalar(170, config.hsv_s_min, config.hsv_v_min), cv::Scalar(180, 255, 255), red_mask2);
        cv::bitwise_or(red_mask1, red_mask2, red_mask);

        int h = red_mask.rows; int split_line = h * 0.5;
        cv::Mat top_roi = red_mask(cv::Rect(0, 0, red_mask.cols, split_line));
        cv::Mat kernel_far = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(config.kernel_far_w, config.kernel_far_h));
        cv::morphologyEx(top_roi, top_roi, cv::MORPH_OPEN, kernel_far);
        cv::Mat bottom_roi = red_mask(cv::Rect(0, split_line, red_mask.cols, h - split_line));
        cv::Mat kernel_near = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(config.kernel_near_w, config.kernel_near_h));
        cv::morphologyEx(bottom_roi, bottom_roi, cv::MORPH_OPEN, kernel_near);

        cv::Mat final_img = dst.clone(); 
        cv::Mat raw_red;
        cv::inRange(src, cv::Scalar(0, 0, 100), cv::Scalar(100, 100, 255), raw_red);
        final_img.setTo(cv::Scalar(0, 255, 0), raw_red);
        final_img.setTo(cv::Scalar(0, 0, 255), red_mask);
        return final_img;
    }

    void fitNonlinearCenterLine(std::vector<cv::Point2f>& pts, const std_msgs::Header& h, cv::Mat& debug_img, const tf::Transform& trans) {
        std::vector<float> y_vals;
        for(const auto& p : pts) y_vals.push_back(p.y);
        std::sort(y_vals.begin(), y_vals.end());
        float mid_y = y_vals[y_vals.size() / 2];
        std::vector<cv::Point2f> left, right;
        for(const auto& p : pts) {
            if (p.y < mid_y) left.push_back(p);
            else right.push_back(p);
        }
        if (left.size() < 15 || right.size() < 15) return;
        std::vector<cv::Point2f> centers;
        float min_x = pts[0].x, max_x = pts[0].x;
        for(auto& p : pts) { min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x); }
        for (float x = min_x; x <= std::min(max_x, min_x + 4.0f); x += 0.2f) {
            float sum_y_l = 0, sum_y_r = 0; int cl = 0, cr = 0;
            for(auto& p : left)  if(std::abs(p.x - x) < 0.25) { sum_y_l += p.y; cl++; }
            for(auto& p : right) if(std::abs(p.x - x) < 0.25) { sum_y_r += p.y; cr++; }
            if (cl > 0 && cr > 0) centers.push_back(cv::Point2f(x, (sum_y_l/cl + sum_y_r/cr)/2.0f));
        }
        if (centers.size() < 6) return;
        cv::Mat A(centers.size(), 3, CV_32F);
        cv::Mat B(centers.size(), 1, CV_32F);
        for(int i=0; i<centers.size(); ++i) {
            A.at<float>(i,0) = pow(centers[i].x, 2); A.at<float>(i,1) = centers[i].x; A.at<float>(i,2) = 1.0;
            B.at<float>(i,0) = centers[i].y;
        }
        cv::Mat coeffs;
        cv::solve(A, B, coeffs, cv::DECOMP_SVD);
        float a = coeffs.at<float>(0), b = coeffs.at<float>(1), c = coeffs.at<float>(2);
        smoothed_a_ = 0.7 * smoothed_a_ + 0.3 * a;
        smoothed_b_ = 0.7 * smoothed_b_ + 0.3 * b;
        smoothed_c_ = 0.7 * smoothed_c_ + 0.3 * c;
        publishMarkerCurve(smoothed_a_, smoothed_b_, smoothed_c_, min_x, h);
    }

    void publishMarkerCurve(float a, float b, float c, float start_x, const std_msgs::Header& h) {
        visualization_msgs::Marker m;
        m.header.frame_id = config.global_frame_id; m.header.stamp = ros::Time::now();
        m.ns = "nav"; m.type = visualization_msgs::Marker::LINE_STRIP;
        m.action = visualization_msgs::Marker::ADD;
        m.id = 0; m.scale.x = 0.1; 
        m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0;
        m.pose.orientation.w = 1.0;
        for (float x = start_x; x <= start_x + 3.5f; x += 0.1f) {
            geometry_msgs::Point p;
            p.x = x; p.y = a*x*x + b*x + c; p.z = 0.5;
            m.points.push_back(p);
        }
        pub_marker_.publish(m);
    }

    void publishMarkers(const std_msgs::Header& header) {
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker;
        marker.header = header;
        marker.ns = "octomap_voxels";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::CUBE_LIST; 
        marker.action = visualization_msgs::Marker::ADD;
        double res = octree_->getResolution();
        marker.scale.x = res; marker.scale.y = res; marker.scale.z = res;
        marker.pose.orientation.w = 1.0;
        for(octomap::ColorOcTree::leaf_iterator it = octree_->begin_leafs(); it != octree_->end_leafs(); ++it) {
            if (octree_->isNodeOccupied(*it)) {
                geometry_msgs::Point p;
                p.x = it.getX(); p.y = it.getY(); p.z = it.getZ();
                marker.points.push_back(p);
                std_msgs::ColorRGBA c;
                c.r = it->getColor().r / 255.0; c.g = it->getColor().g / 255.0; c.b = it->getColor().b / 255.0; c.a = 1.0;
                marker.colors.push_back(c);
            }
        }
        if (!marker.points.empty()) {
            marker_array.markers.push_back(marker);
            pub_vis_markers_.publish(marker_array);
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_lidar_;
    message_filters::Subscriber<sensor_msgs::Image> *sub_depth_filter_;
    message_filters::Subscriber<sensor_msgs::Image> *sub_rgb_filter_;
    message_filters::Synchronizer<MySyncPolicy> *sync_;
    ros::Publisher pub_octomap_, pub_2d_map_, pub_marker_, pub_debug_img_, pub_vp_debug_, pub_vis_markers_;
    tf::TransformListener tf_listener_;
    octomap::ColorOcTree* octree_;
    float smoothed_a_, smoothed_b_, smoothed_c_;
    StructureOdometer semantic_odom_; 
    std::unordered_map<octomap::OcTreeKey, float, octomap::OcTreeKey::KeyHash> semantic_probs_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "semantic_octomap_node");
    SemanticOctomapServer server;
    ros::spin();
    return 0;
}