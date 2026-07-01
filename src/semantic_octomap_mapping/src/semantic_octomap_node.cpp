#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>               
#include <nav_msgs/Odometry.h>           
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h> 
#include <visualization_msgs/MarkerArray.h>
#include <gtsam/base/numericalDerivative.h> 
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h> 
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
// 在文件最上面的 include 区域添加这行
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/pca.h> 
#include <pcl_ros/transforms.h> 
#include <pcl/filters/radius_outlier_removal.h> 

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>    
#include <deque>  
#include <cmath>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <shared_mutex> // [新增] C++17 读写锁
#include <numeric>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Unit3.h> // [新增] 引入单位球面流形几何
#include <gtsam/geometry/Cal3_S2.h>       
#include <gtsam/geometry/PinholeCamera.h> 
#include <gtsam/geometry/OrientedPlane3.h> 
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/linear/NoiseModel.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <omp.h>
#include <fstream>
#include <sys/stat.h>
#include <iomanip>
#include <chrono>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
using namespace gtsam;



// ================= [🔴 请复制并替换整个 DataLogger 类] =================

// [新增] 全局数据记录器类 (包含所有实验数据接口 - 最终完整版)
class DataLogger {
    public:
        // 1. 基础实验数据
        std::ofstream f_traj;        // 轨迹 & 速度 (Fig. 8, Table 3)
        std::ofstream f_slip;        // 滑移参数 & 机动强度 (Section 2.3.1)
        std::ofstream f_eigen;       // 退化特征值 & 约束硬度 (Section 2.3.4)
        std::ofstream f_feature;     // 前端特征稳定性 (Section 3.2)
        std::ofstream f_time;        // 耗时分析 (Table 6)
        std::ofstream f_calib;       // 在线标定收敛 (Section 2.2.9)
        
        // 2. 高级分析数据 (最新补充)
        std::ofstream f_map_stats;   // 地图生长 & 动态剔除统计 (Section 2.4.2)
        std::ofstream f_graph_error; // 因子图总误差 (收敛性分析)
        std::ofstream f_flow_stats;  // 视觉光流权重自适应 (Section 2.3.5)

        std::ofstream f_leg_idx;
    
        // 数据保存路径
        std::string base_path = "/home/jiangjiacheeng/spm_data/"; 
    
        DataLogger() {
            // 创建目录
            mkdir(base_path.c_str(), 0777);
    
            // 打开所有文件流
            f_traj.open(base_path + "trajectory.txt", std::ios::out);
            f_slip.open(base_path + "slip_params.txt", std::ios::out);
            f_eigen.open(base_path + "eigen_stats.txt", std::ios::out);
            f_feature.open(base_path + "features.txt", std::ios::out);
            f_time.open(base_path + "runtime.txt", std::ios::out);
            f_calib.open(base_path + "calibration.txt", std::ios::out);
            f_map_stats.open(base_path + "map_stats.txt", std::ios::out);
            f_graph_error.open(base_path + "graph_error.txt", std::ios::out);
            f_flow_stats.open(base_path + "flow_stats.txt", std::ios::out);
            f_leg_idx.open(base_path + "leg_indices.txt", std::ios::out);
            
            // ================== [写入详细表头] ==================
            
            // 轨迹: 时间, 位置(x,y,z), 姿态(qx,qy,qz,qw), 速度(vx,vy,vz)
            f_traj << "time x y z qx qy qz qw vx vy vz" << std::endl;
            
            // 滑移: 时间, 滑移系数(kv, kw), 动态方差, 控制指令, 是否转向
            f_slip << "time kappa_v kappa_omega sigma_slip_v sigma_slip_w cmd_v cmd_w is_turning" << std::endl; 
            
            // [🔴 修改] 特征值表头增加了 'constraint_sigma' (约束硬度)
            f_eigen << "time lambda_max_cov is_degenerate structure_lock_active constraint_sigma" << std::endl;
            
            // 特征: 时间, 周期性间距, 置信度, VP Yaw, VP内点, VP有效性, 走廊角度, 线特征有效性
            f_feature << "time period_spacing period_conf vp_yaw vp_inliers vp_valid aisle_angle line_valid" << std::endl;
            
            // 耗时: 时间, 预处理, 视觉前端, 雷达前端, 后端优化, 建图, 总耗时
            f_time << "time t_pre t_vis t_lidar t_opt t_map t_total" << std::endl;
            
            // 标定: 时间, Yaw偏置, Pitch偏置
            f_calib << "time yaw_bias_deg pitch_bias_deg" << std::endl;
    
            // [🔴 修改] 地图统计增加了 'dynamic_points_rejected' (动态剔除数)
            f_map_stats << "time node_count memory_usage_mb dynamic_points_rejected static_points_inserted" << std::endl;
    
            // 图误差: 时间, 总误差(Chi2), 归一化误差
            f_graph_error << "time total_error error_norm" << std::endl;
            
            // [🔴 新增] 光流统计: 时间, 视觉速度, 预测速度, 置信度, 深度, 计算出的Sigma
            f_flow_stats << "time vis_vel_x pred_vel_x confidence avg_depth calculated_sigma weight_active" << std::endl;

            f_leg_idx << "time leg_index predicted_dist current_spacing is_snap_event" << std::endl;
            
            ROS_INFO("DataLogger Initialized. All 9 log files active at %s", base_path.c_str());
        }
    
        ~DataLogger() {
            if(f_traj.is_open()) f_traj.close();
            if(f_slip.is_open()) f_slip.close();
            if(f_eigen.is_open()) f_eigen.close();
            if(f_feature.is_open()) f_feature.close();
            if(f_time.is_open()) f_time.close();
            if(f_calib.is_open()) f_calib.close();
            if(f_map_stats.is_open()) f_map_stats.close();
            if(f_graph_error.is_open()) f_graph_error.close();
            if(f_flow_stats.is_open()) f_flow_stats.close();
            if(f_leg_idx.is_open()) f_leg_idx.close();
        }
    };
    
    // 全局实例
    DataLogger g_logger;
    
    // ================= [🔴 替换结束] =================

struct SystemConfig {
    double leg_spacing;
    double lidar_z_min; 
    double lidar_z_max; 
    double z_ceiling;        
    double z_obstacle_thresh;
    double ground_offset_k;
    double camera_z_shift; 
    double hole_fill_area;
    int hsv_s_min; int hsv_v_min;
    int kernel_far_w, kernel_far_h;
    int kernel_near_w, kernel_near_h;
    double roi_h_ratio; double roi_w_ratio;
    double trigger_thresh;
    double cooldown_time;
    double outlier_thresh;      
    double semantic_hit_prob;   
    double semantic_miss_prob;  
    double semantic_thresh;     
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

    double odom_slip_gain_rot;
    double odom_slip_gain_trans;
    double odom_min_sigma_y;
    
    double cam_fx, cam_fy, cam_cx, cam_cy; 
    double perspective_lookahead;    

    bool enable_auto_calibration;
    double calib_learning_rate;   

    double intensity_ref_dist;  
    double intensity_alpha;     
    double geometry_check_radius; 
    double linearity_thresh;      

    double dynamic_dist_diff; 

    int flow_min_pts;
    double flow_quality;
    double flow_min_dist;
    double flow_ransac_thresh;
    double flow_weight;

    double noise_odom_v_base;
    double noise_odom_w_base;
    double noise_flow_base;

    int temporal_hit_thresh;
    double temporal_static_life;

    // [新增] 自适应滑移学习参数
    double slip_learn_sigma_base;    // 静态或直行时的滑移参数波动率
    double slip_learn_sigma_gain;    // 转向时的滑移参数不确定性增益

    // [新增 - 修复报错] 必须添加以下缺失参数
    double degeneracy_lock_sigma;   // 退化时的锁定强度
    int temporal_trust_thresh;      // 信任计数阈值 (多视角验证)

    double vp_min_activation_dist;

    bool enable_rail_mode;
};

SystemConfig config;

void loadParameters(ros::NodeHandle& nh) {
    nh.param<double>("leg_spacing", config.leg_spacing, 1.66);
    nh.param<double>("lidar_z_min", config.lidar_z_min, 0.2); 
    nh.param<double>("lidar_z_max", config.lidar_z_max, 2.5); 
    nh.param<double>("z_ceiling", config.z_ceiling, 3.0); 
    nh.param<double>("z_obstacle_thresh", config.z_obstacle_thresh, 2.05);
    nh.param<double>("ground_offset_k", config.ground_offset_k, 0.35);
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
    nh.param<double>("outlier_thresh", config.outlier_thresh, 5.0);     
    nh.param<double>("semantic_hit_prob", config.semantic_hit_prob, 0.2); 
    nh.param<double>("semantic_miss_prob", config.semantic_miss_prob, 0.05); 
    nh.param<double>("semantic_thresh", config.semantic_thresh, 0.75); 
    nh.param<double>("dust_intensity_thresh", config.dust_intensity_thresh, 0.1);
    nh.param<double>("prob_miss_static", config.prob_miss_static, 0.46); 
    nh.param<double>("prob_miss_dynamic", config.prob_miss_dynamic, 0.35); 
    nh.param<double>("vp_weight_base", config.vp_weight_base, 50); 

    nh.param<double>("odom_slip_gain_rot", config.odom_slip_gain_rot, 3.0);
    nh.param<double>("odom_slip_gain_trans", config.odom_slip_gain_trans, 5.0);
    nh.param<double>("odom_min_sigma_y", config.odom_min_sigma_y, 0.05);
    nh.param<std::string>("global_frame_id", config.global_frame_id, "camera_init"); 

    nh.param<double>("cam_fx", config.cam_fx, 904.3275);
    nh.param<double>("cam_fy", config.cam_fy, 903.9418);
    nh.param<double>("cam_cx", config.cam_cx, 653.1630);
    nh.param<double>("cam_cy", config.cam_cy, 369.6624);
    nh.param<double>("perspective_lookahead", config.perspective_lookahead, 40.0);
    
    nh.param<bool>("enable_auto_calibration", config.enable_auto_calibration, false);
    nh.param<bool>("enable_rail_mode", config.enable_rail_mode, false);
    ROS_INFO("Config Loaded. Rail Mode: %s", config.enable_rail_mode ? "ON" : "OFF");
    nh.param<double>("calib_learning_rate", config.calib_learning_rate, 0.0005); 

    nh.param<double>("intensity_ref_dist", config.intensity_ref_dist, 3.0);
    nh.param<double>("intensity_alpha", config.intensity_alpha, 2.0); 
    nh.param<double>("geometry_check_radius", config.geometry_check_radius, 0.40); 
    nh.param<double>("linearity_thresh", config.linearity_thresh, 0.4); 

    nh.param<double>("dynamic_dist_diff", config.dynamic_dist_diff, 0.50); 

    nh.param<int>("flow_min_pts", config.flow_min_pts, 30);
    nh.param<double>("flow_quality", config.flow_quality, 0.01);
    nh.param<double>("flow_min_dist", config.flow_min_dist, 20.0);
    nh.param<double>("flow_ransac_thresh", config.flow_ransac_thresh, 0.1);
    nh.param<double>("flow_weight", config.flow_weight, 5.0); 

    nh.param<double>("noise_odom_v_base", config.noise_odom_v_base, 0.05);
    nh.param<double>("noise_odom_w_base", config.noise_odom_w_base, 0.2);
    nh.param<double>("noise_flow_base", config.noise_flow_base, 0.1);      
    
    auto probToLogOdds = [](double p) { return std::log(p / (1.0 - p)); };
    config.log_odds_miss_static = probToLogOdds(config.prob_miss_static);
    config.log_odds_miss_dynamic = probToLogOdds(config.prob_miss_dynamic);
    config.log_odds_hit = probToLogOdds(0.7);      
    config.log_odds_min = probToLogOdds(0.12);      
    config.log_odds_max = probToLogOdds(0.97);      

    nh.param<int>("temporal_hit_thresh", config.temporal_hit_thresh, 3);
    nh.param<double>("temporal_static_life", config.temporal_static_life, 2.0);

    // base: 1e-4 表示在直行时我们非常信任之前的滑移估计，不轻易改变
    nh.param<double>("slip_learn_sigma_base", config.slip_learn_sigma_base, 1e-4); 
    // gain: 0.01 表示每增加 1rad/s 的转向，滑移参数的不确定性增加 0.01
    nh.param<double>("slip_learn_sigma_gain", config.slip_learn_sigma_gain, 0.01);
    // [新增 - 修复报错] 加载缺失参数
    // 当检测到退化时，强行将周期性约束的 sigma 压到极小 (0.01)，相当于权重极大，锁死位置
    nh.param<double>("degeneracy_lock_sigma", config.degeneracy_lock_sigma, 0.01);
    
    // 只有当障碍物在至少 2 个不同的位置被观测到时，才认为是可信的 (防止单帧噪点)
    nh.param<int>("temporal_trust_thresh", config.temporal_trust_thresh, 2);

    nh.param<double>("vp_min_activation_dist", config.vp_min_activation_dist, 2.0);
    ROS_INFO("Config v70.1 (Adaptive Slip) Loaded.");
}

class ExtrinsicCalibrator {
    private:
        double yaw_bias_;   
        double pitch_bias_; 
        int samples_count_;
        int success_search_count_;
        const double MAX_BIAS = 10.0 * M_PI / 180.0; 
    
    public:
        ExtrinsicCalibrator() : yaw_bias_(0.27 * M_PI / 180.0), pitch_bias_(0.0), samples_count_(0), success_search_count_(0) {}
    
        void update(int u, int v, const cv::Mat& mask, cv::Mat& debug_img) {
            if (!config.enable_auto_calibration) return;
            if (u < 20 || u >= mask.cols - 20 || v < 20 || v >= mask.rows - 20) return;
            samples_count_++;
            if (mask.at<uchar>(v, u) > 128) {
                cv::circle(debug_img, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), -1);
                return; 
            }
            int search_range = 100; 
            int best_du = 0;
            int min_dist_u = 9999;
            for (int du = -search_range; du <= search_range; du += 2) {
                int u_new = u + du;
                if (u_new >= 0 && u_new < mask.cols) {
                    if (mask.at<uchar>(v, u_new) > 128) { 
                        if (std::abs(du) < min_dist_u) {
                            min_dist_u = std::abs(du);
                            best_du = du;
                        }
                    }
                }
            }
            int best_dv = 0;
            int min_dist_v = 9999;
            if (min_dist_u < 50) { 
                for (int dv = -search_range; dv <= search_range; dv += 2) {
                    int v_new = v + dv;
                    if (v_new >= 0 && v_new < mask.rows) {
                        int target_u = u + best_du; 
                        if (target_u >= 0 && target_u < mask.cols) {
                            if (mask.at<uchar>(v_new, target_u) > 128) {
                                if (std::abs(dv) < min_dist_v) {
                                    min_dist_v = std::abs(dv);
                                    best_dv = dv;
                                }
                            }
                        }
                    }
                }
            }
            if (min_dist_u < 9999 || min_dist_v < 9999) {
                success_search_count_++;
                if(min_dist_u < 9999)
                    cv::line(debug_img, cv::Point(u, v), cv::Point(u + best_du, v), cv::Scalar(255, 255, 255), 1);
                if(min_dist_v < 9999)
                    cv::line(debug_img, cv::Point(u + best_du, v), cv::Point(u + best_du, v + best_dv), cv::Scalar(0, 255, 255), 1);
                double learning_rate = config.calib_learning_rate * 2.0; 
                if (min_dist_u < 9999) yaw_bias_   += learning_rate * (best_du * 0.001);
                if (min_dist_v < 9999) pitch_bias_ -= learning_rate * (best_dv * 0.001); 
                yaw_bias_ = std::max(-MAX_BIAS, std::min(MAX_BIAS, yaw_bias_));
                pitch_bias_ = std::max(-MAX_BIAS, std::min(MAX_BIAS, pitch_bias_));
            }
        }
    
        gtsam::Rot3 getCorrectionRotation() {
            return gtsam::Rot3::Ypr(yaw_bias_, pitch_bias_, 0.0);
        }

        // [🔴 新增代码] 获取当前偏置值用于记录
        std::pair<double, double> getCurrentBias() {
            return {yaw_bias_, pitch_bias_};
        }
        
        std::string getStatusString() {
            double yaw_deg = yaw_bias_ * 180.0 / M_PI;
            double pitch_deg = pitch_bias_ * 180.0 / M_PI;
            char buffer[100];
            sprintf(buffer, "Y:%.2f P:%.2f", yaw_deg, pitch_deg);
            return std::string(buffer);
        }
    
        void printDiagnostics() {
            static double last_print = 0;
            if (ros::Time::now().toSec() - last_print > 5.0) { 
                ROS_INFO("[Calib Status] Samples:%d, Links:%d | Bias(deg) Yaw:%.4f Pitch:%.4f", 
                         samples_count_, success_search_count_, 
                         yaw_bias_ * 180.0 / M_PI, pitch_bias_ * 180.0 / M_PI);
                samples_count_ = 0;
                success_search_count_ = 0;
                last_print = ros::Time::now().toSec();
            }
        }

        void updateJointly(double visual_yaw_rad, double lidar_angle_rad) {
            if (!config.enable_auto_calibration) return;
            double diff = lidar_angle_rad - visual_yaw_rad;
            while (diff > M_PI) diff -= 2 * M_PI;
            while (diff < -M_PI) diff += 2 * M_PI;
            if (std::abs(diff) > 0.008) return; 
            yaw_bias_ = (1.0 - 0.005) * yaw_bias_ + 0.005 * diff;
            yaw_bias_ = std::max(-MAX_BIAS, std::min(MAX_BIAS, yaw_bias_));
        }
    };

class PeriodicityEstimator {
    private:
        std::deque<float> local_x_history_; 
        double estimated_spacing_;
        double confidence_;
        const double BIN_SIZE = 0.02;     
        const double WINDOW_SIZE = 6.0;   
        const double GRID_RES = 0.05;     
    
    public:
        PeriodicityEstimator() : estimated_spacing_(1.66), confidence_(0.0) {}
    

        // [🔴 新增] 手动重置函数
        void reset() {
            local_x_history_.clear();
            estimated_spacing_ = 1.66; // 恢复初始值
            confidence_ = 0.0;         // 恢复初始置信度
        }
        void addPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
                const gtsam::Pose3& current_pose, 
                const cv::Mat& visual_mask,
                const gtsam::Pose3& world_to_cam, 
                const gtsam::Cal3_S2& K) {
            if (cloud->empty()) return;
            struct GridCell { 
                float min_y; float max_y; 
                bool vis_confirmed; 
                std::vector<gtsam::Point3> points; 
            };
            std::unordered_map<int, GridCell> grid_stats; 
            for (const auto& p : cloud->points) {
                gtsam::Point3 p_w(p.x, p.y, p.z);
                gtsam::Point3 p_c;
                try {
                    p_c = world_to_cam.transformTo(p_w);
                } catch(...) { continue; }
                if (p_c.z() < 0.2 || p_c.z() > 8.0) continue;
                if (std::abs(p_c.y()) > 4.0) continue;
                bool is_visually_confirmed = false;
                int u = std::round(p_c.x() * K.fx() / p_c.z() + K.px());
                int v = std::round(p_c.y() * K.fy() / p_c.z() + K.py());
                if (!visual_mask.empty()) {
                    if (u >= 0 && u < visual_mask.cols && v >= 0 && v < visual_mask.rows) {
                        if (visual_mask.at<uchar>(v, u) > 128) { 
                            is_visually_confirmed = true;
                        }
                    }
                }
                int x_idx = std::round(p.x / GRID_RES); 
                int key = x_idx; 
                if (grid_stats.find(key) == grid_stats.end()) {
                    grid_stats[key] = { (float)p_c.y(), (float)p_c.y(), is_visually_confirmed, {p_c} };
                } else {
                    if (p_c.y() < grid_stats[key].min_y) grid_stats[key].min_y = p_c.y();
                    if (p_c.y() > grid_stats[key].max_y) grid_stats[key].max_y = p_c.y();
                    if (is_visually_confirmed) grid_stats[key].vis_confirmed = true;
                    grid_stats[key].points.push_back(p_c); 
                }
            }
            for (const auto& kv : grid_stats) {
                float y_span = kv.second.max_y - kv.second.min_y; 
                bool vis_pass = kv.second.vis_confirmed;
                int point_count = kv.second.points.size();
                bool accepted = false;
                
                if (vis_pass && point_count >= 1) accepted = true;
                if (y_span > 0.03 && point_count >= 2) accepted = true;
                if (point_count >= 3 && y_span > 0.02) accepted = true;
                
                if (accepted) {
                    float recovered_x = kv.first * GRID_RES;
                    local_x_history_.push_back(recovered_x);
                    if (vis_pass) local_x_history_.push_back(recovered_x); 
                }
            }
            if (local_x_history_.size() > 6000) { 
                local_x_history_.erase(local_x_history_.begin(), local_x_history_.begin() + 1500);
            }
        }
    
        void updateEstimate() {
            if (local_x_history_.empty()) return;
            float min_x = *std::min_element(local_x_history_.begin(), local_x_history_.end());
            float max_x = *std::max_element(local_x_history_.begin(), local_x_history_.end());
            int num_bins = std::ceil((max_x - min_x) / BIN_SIZE);
            if (num_bins < 15) return; 
            
            std::vector<int> histogram(num_bins, 0);
            for (float x : local_x_history_) {
                int idx = (x - min_x) / BIN_SIZE;
                if (idx >= 0 && idx < num_bins) histogram[idx]++;
            }

            auto calc_correlation = [&](int lag) -> double {
                if (lag < 1 || lag >= num_bins) return 0.0;
                double sum = 0.0;
                int overlap_count = 0;
                for (int i = 0; i < num_bins - lag; ++i) {
                    sum += (histogram[i] * histogram[i + lag]);
                    overlap_count++;
                }
                return (overlap_count > 0) ? (sum / overlap_count) : 0.0;
            };

            int min_lag_idx = 1.5 / BIN_SIZE;
            int max_lag_idx = 1.8 / BIN_SIZE;
            
            double max_correlation = -1.0;
            int best_lag = 0;

            for (int lag = min_lag_idx; lag <= max_lag_idx; ++lag) {
                double score = calc_correlation(lag);
                if (score > max_correlation) {
                    max_correlation = score;
                    best_lag = lag;
                }
            }

            if (max_correlation > 0.30) { 
                double measured_lag = (double)best_lag;
                if (best_lag > min_lag_idx && best_lag < max_lag_idx) {
                    double val_center = max_correlation;
                    double val_left   = calc_correlation(best_lag - 1);
                    double val_right  = calc_correlation(best_lag + 1);

                    double denominator = val_left - 2.0 * val_center + val_right;
                    if (std::abs(denominator) > 1e-5) { 
                        double delta = 0.5 * (val_left - val_right) / denominator;
                        delta = std::max(-0.5, std::min(0.5, delta)); 
                        measured_lag += delta;
                    }
                }
                double measured_dist = measured_lag * BIN_SIZE;
                estimated_spacing_ = 0.95 * estimated_spacing_ + 0.05 * measured_dist;
                confidence_ = std::min(1.0, confidence_ + 0.1);
            } else {
                confidence_ = std::max(0.0, confidence_ - 0.01);
            }
        }
    
        double getSpacing() const { return estimated_spacing_; }
        double getConfidence() const { return confidence_; }
}; 

class TrackKinematicsFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector2> {
private:
    double measured_v_;      
    double measured_omega_;  
    double dt_;              

public:
    TrackKinematicsFactor(gtsam::Key pose_i, gtsam::Key pose_j, gtsam::Key calib_key, 
                          double v, double omega, double dt, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector2>(model, pose_i, pose_j, calib_key),
          measured_v_(v), measured_omega_(omega), dt_(dt) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose_i, const gtsam::Pose3& pose_j, 
                                const gtsam::Vector2& calib, 
                                boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none,
                                boost::optional<gtsam::Matrix&> H3 = boost::none) const override {
        
        double scale_v = calib(0);       
        double scale_omega = calib(1);   

        double v_eff = measured_v_ * scale_v;
        double omega_eff = measured_omega_ * scale_omega;

        double dx = v_eff * dt_;
        double dtheta = omega_eff * dt_;
        
        gtsam::Rot3 rot_pred = gtsam::Rot3::Ypr(dtheta, 0.0, 0.0);
        gtsam::Point3 trans_pred(dx, 0.0, 0.0); 
        gtsam::Pose3 T_pred(rot_pred, trans_pred);

        gtsam::Pose3 T_rel = pose_i.between(pose_j); 
        gtsam::Vector error = gtsam::traits<gtsam::Pose3>::Local(T_pred, T_rel);

        if (H1) *H1 = gtsam::numericalDerivative31<gtsam::Vector, gtsam::Pose3, gtsam::Pose3, gtsam::Vector2>(
            boost::bind(&TrackKinematicsFactor::evaluateError, this, _1, _2, _3, boost::none, boost::none, boost::none), 
            pose_i, pose_j, calib);
            
        if (H2) *H2 = gtsam::numericalDerivative32<gtsam::Vector, gtsam::Pose3, gtsam::Pose3, gtsam::Vector2>(
            boost::bind(&TrackKinematicsFactor::evaluateError, this, _1, _2, _3, boost::none, boost::none, boost::none), 
            pose_i, pose_j, calib);
            
        if (H3) *H3 = gtsam::numericalDerivative33<gtsam::Vector, gtsam::Pose3, gtsam::Pose3, gtsam::Vector2>(
            boost::bind(&TrackKinematicsFactor::evaluateError, this, _1, _2, _3, boost::none, boost::none, boost::none), 
            pose_i, pose_j, calib);

        return error;
    }
};

class StructurePeriodicityFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    double spacing_;
public:
    StructurePeriodicityFactor(gtsam::Key key, double spacing, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), spacing_(spacing) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        double x = pose.x();
        double k = std::round(x / spacing_);
        double target = k * spacing_;
        double error = x - target;

        if (H) {
             *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
                boost::bind(&StructurePeriodicityFactor::evaluateError, this, _1, boost::none), pose);
        }
        return gtsam::Vector1(error);
    }
};

class VanishingPointFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    double measured_yaw_; 
public:
    VanishingPointFactor(gtsam::Key poseKey, double measured_yaw, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey), measured_yaw_(measured_yaw) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (H) {
            *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
                boost::bind(&VanishingPointFactor::evaluateError, this, _1, boost::none), pose);
        }
        Rot3 R = pose.rotation();
        double predicted_yaw = R.yaw();
        double error = predicted_yaw - measured_yaw_;
        while (error > M_PI) error -= 2 * M_PI;
        while (error < -M_PI) error += 2 * M_PI;
        return gtsam::Vector1(error);
    }
};

class CorridorPerspectiveFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    gtsam::Point2 measured_uv_; 
    gtsam::Cal3_S2::shared_ptr K_; 
    double lookahead_dist_;     

public:
    CorridorPerspectiveFactor(gtsam::Key poseKey, gtsam::Point2 uv, gtsam::Cal3_S2::shared_ptr K, double dist, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey), measured_uv_(uv), K_(K), lookahead_dist_(dist) {}

        gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
            gtsam::Point3 dir_world(1.0, 0.0, 0.0); 
            gtsam::Rot3 wRb = pose.rotation();
            gtsam::Point3 dir_body = wRb.unrotate(dir_world); 
            gtsam::Point3 dir_cam(-dir_body.y(), -dir_body.z(), dir_body.x());
    
            if (dir_cam.z() < 1e-3) {
                if (H) *H = gtsam::Matrix::Zero(2, 6);
                return gtsam::Vector2(0.0, 0.0);
            }
    
            double u_pred = dir_cam.x() / dir_cam.z() * K_->fx() + K_->px();
            double v_pred = dir_cam.y() / dir_cam.z() * K_->fy() + K_->py();
            gtsam::Vector2 error(u_pred - measured_uv_.x(), v_pred - measured_uv_.y());
    
            if (H) {
                gtsam::Matrix H_full = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
                    boost::bind(&CorridorPerspectiveFactor::evaluateError, this, _1, boost::none), pose);
                H_full.block<2, 3>(0, 3).setZero(); 
                *H = H_full;
            }
            return error;
        }
};


// [修复版] Manifold VP Factor - 支持双向走廊
// 自动检测并适配 +X 或 -X 方向的行进
class ManifoldVPFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    private:
        gtsam::Unit3 measured_direction_cam_; 
        gtsam::Pose3 bRc_;                    
    
    public:
        ManifoldVPFactor(gtsam::Key key, 
                         const cv::Point2f& uv_measurement,
                         const gtsam::Cal3_S2& K,
                         const gtsam::Pose3& body_to_cam,
                         gtsam::SharedNoiseModel model)
            : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), 
              bRc_(body_to_cam) {
            
            // 将像素坐标转换为相机坐标系下的单位向量
            double nx = (uv_measurement.x - K.px()) / K.fx();
            double ny = (uv_measurement.y - K.py()) / K.fy();
            measured_direction_cam_ = gtsam::Unit3(gtsam::Vector3(nx, ny, 1.0));
            // 注意：不再在这里写死 world_axis_，移到 evaluateError 中动态判断
        }
    
        gtsam::Vector evaluateError(const gtsam::Pose3& pose_w_b, 
                                    boost::optional<gtsam::Matrix&> H = boost::none) const override {
            try {
                // 1. 计算相机在世界坐标系下的位姿
                gtsam::Pose3 pose_w_c = pose_w_b.compose(bRc_);
                
                // 2. 定义走廊的主轴 (全局 X 轴)
                gtsam::Unit3 world_axis_pos(1.0, 0.0, 0.0);
                
                // 3. 将全局 X 轴投影到当前相机坐标系 (如果在 +X 方向行驶时的预测灭点)
                gtsam::Unit3 predicted_cam = pose_w_c.rotation().unrotate(world_axis_pos, boost::none); 

                // 4. [修复核心] 检查方向对齐情况
                // 如果点积为负，说明我们正朝向 -X 方向行驶。
                // 在走廊中，无论朝东还是朝西，只要平行于轴线，灭点约束都应有效。
                double dot = measured_direction_cam_.unitVector().dot(predicted_cam.unitVector());
                
                if (dot < 0.0) {
                    // 我们正背对 +X 轴。
                    // 将预测向量反转，使其指向相机坐标系的“前方”，
                    // 这样才能正确计算与测量值（也在前方）的误差。
                    predicted_cam = gtsam::Unit3(-predicted_cam.unitVector());
                    dot = -dot; // 将点积转为正值，用于下方的安全检查
                }
                
                // 5. 安全检查：如果反转后偏差依然过大（例如面朝墙壁），则忽略
                if (dot < 0.2) { // 稍微提高阈值到 0.2 (~78度) 以增加安全性
                    if (H) *H = gtsam::Matrix::Zero(2, 6);
                    return gtsam::Vector2::Zero();
                }
    
                // 6. 计算数值微分 (Jacobian)
                if (H) {
                    *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
                        boost::bind(&ManifoldVPFactor::evaluateError, this, _1, boost::none), pose_w_b, 1e-5);
                    
                    // [核弹级修复] 雅可比矩阵 NaN/Inf 检查
                    bool is_bad_jacobian = false;
                    for(int i=0; i<H->rows(); ++i) {
                        for(int j=0; j<H->cols(); ++j) {
                            if (!std::isfinite((*H)(i,j)) || std::abs((*H)(i,j)) > 1e5) {
                                is_bad_jacobian = true;
                                break;
                            }
                        }
                    }
    
                    if (is_bad_jacobian) {
                        *H = gtsam::Matrix::Zero(2, 6);
                        return gtsam::Vector2::Zero();
                    }
                }
    
                // 7. 计算误差
                gtsam::Vector2 error = measured_direction_cam_.errorVector(predicted_cam);
                
                // 输出结果再次检查
                if (!std::isfinite(error[0]) || !std::isfinite(error[1])) {
                    if (H) *H = gtsam::Matrix::Zero(2, 6);
                    return gtsam::Vector2::Zero();
                }
    
                return error;
    
            } catch (...) {
                // 捕获所有几何异常
                if (H) *H = gtsam::Matrix::Zero(2, 6);
                return gtsam::Vector2::Zero();
            }
        }
};
class ManhattanHeadingFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
public:
    ManhattanHeadingFactor(gtsam::Key key, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        double current_yaw = pose.rotation().yaw();
        double k = std::round(current_yaw / M_PI_2);
        double target_yaw = k * M_PI_2;
        
        double error = current_yaw - target_yaw;
        while (error > M_PI) error -= 2 * M_PI;
        while (error < -M_PI) error += 2 * M_PI;

        if (H) {
             *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
                boost::bind(&ManhattanHeadingFactor::evaluateError, this, _1, boost::none), pose);
        }
        return gtsam::Vector1(error);
    }
};

class PlanarConstraintFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    double z_target_;
public:
    PlanarConstraintFactor(gtsam::Key key, double z_target, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), z_target_(z_target) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (H) {
            *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
                boost::bind(&PlanarConstraintFactor::evaluateError, this, _1, boost::none), pose);
        }
        double z_err = pose.z() - z_target_;
        double roll_err = pose.rotation().roll();
        double pitch_err = pose.rotation().pitch();
        return (gtsam::Vector(3) << roll_err, pitch_err, z_err).finished();
    }
};

class GroundPlaneFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
private:
    gtsam::Vector4 measured_plane_; 

public:
    GroundPlaneFactor(gtsam::Key key, const gtsam::Vector4& measured_plane_coeffs, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), measured_plane_(measured_plane_coeffs) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        gtsam::OrientedPlane3 world_ground(0.0, 0.0, 1.0, 0.0);
        gtsam::Matrix H_transform; 
        gtsam::OrientedPlane3 predicted_plane_in_body = world_ground.transform(pose.inverse(), boost::none, H_transform);
        gtsam::Vector4 pred_vec = predicted_plane_in_body.planeCoefficients();
        if (H) {
             *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
                boost::bind(&GroundPlaneFactor::evaluateError, this, _1, boost::none), pose);
        }
        return pred_vec - measured_plane_;
    }
};

// [🔴 核心升级] 语义线特征因子：同时锁定 航向角(Yaw) 和 横向位置(Y)
class SemanticLineFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    private:
        double measured_angle_; 
        double measured_dist_;  // 机器人到直线的垂直距离
        double line_global_y_;  // 直线在世界坐标系下的理论 Y 值 (锚点)
    
    public:
        SemanticLineFactor(gtsam::Key key, double measured_angle, double measured_dist, double line_global_y, gtsam::SharedNoiseModel model)
            : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), 
              measured_angle_(measured_angle), measured_dist_(measured_dist), line_global_y_(line_global_y) {}
    
        gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
            // 1. 角度误差 (原有逻辑)
            double global_yaw = pose.rotation().yaw();
            double line_angle_world = global_yaw + measured_angle_;
            double error_angle = std::sin(line_angle_world); // 假设墙是平行于 X 轴的

            // 2. [新增] 横向距离误差
            // 逻辑：机器人的 Y 坐标 + (方向符号 * 测量距离) 应该等于 墙的全局 Y
            // 这里我们需要判断墙在机器人的左边还是右边。
            // 简化假设：如果在走廊正中，measure_dist 是正值。
            // 我们构建一个误差： |PoseY - GlobalWallY| - MeasuredDist
            double dist_error = (std::abs(pose.y() - line_global_y_) - measured_dist_);

            if (H) {
                gtsam::Matrix H_pose = gtsam::Matrix::Zero(2, 6);
                
                // 角度雅可比
                H_pose(0, 2) = std::cos(line_angle_world); 
                
                // 距离雅可比 (对 Pose Y 的导数)
                // d(abs(y - y_wall))/dy = sign(y - y_wall)
                double sign = (pose.y() > line_global_y_) ? 1.0 : -1.0;
                H_pose(1, 4) = sign; 

                *H = H_pose;
            }
            return gtsam::Vector2(error_angle, dist_error);
        }
};

class StructuralFeatureExtractor {
    public:
        StructuralFeatureExtractor() {}
    
        std::pair<bool, gtsam::Vector4> extractGroundPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
            if (cloud->points.size() < 100) return {false, gtsam::Vector4::Zero()};
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZI> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.05); 
            seg.setMaxIterations(100);
            pcl::PointCloud<pcl::PointXYZI>::Ptr ground_candidates(new pcl::PointCloud<pcl::PointXYZI>());
            for(const auto& p : cloud->points) {
                if(p.z < 0.5 && p.z > -0.5) ground_candidates->points.push_back(p);
            }
            if (ground_candidates->empty()) return {false, gtsam::Vector4::Zero()};
            seg.setInputCloud(ground_candidates);
            seg.segment(*inliers, *coefficients);
            if (inliers->indices.empty()) return {false, gtsam::Vector4::Zero()};
            float a = coefficients->values[0];
            float b = coefficients->values[1];
            float c = coefficients->values[2];
            float d = coefficients->values[3];
            if (std::abs(c) < 0.85) return {false, gtsam::Vector4::Zero()}; 
            if (c < 0) { a = -a; b = -b; c = -c; d = -d; }
            return {true, gtsam::Vector4(a, b, c, d)};
        }
    
        bool extractVerticalPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, double dist_thresh = 0.1) {
            if (cloud->points.size() < 50) return false;
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZI> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE); 
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0)); 
            seg.setEpsAngle(10.0 * M_PI / 180.0);       
            seg.setDistanceThreshold(dist_thresh); 
            seg.setMaxIterations(100);
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coefficients);
            if (inliers->indices.size() > 200) return true;
            return false;
        }
    
        struct LineResult {
            bool valid;
            double angle;      
            double distance;   
            int inliers;       
        };

        LineResult extractCorridorLine(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
            LineResult res = {false, 0.0, 0.0, 0};
            if (cloud->points.size() < 10) return res;

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_flat(new pcl::PointCloud<pcl::PointXYZI>());
            for (const auto& p : cloud->points) {
                pcl::PointXYZI p_flat = p;
                p_flat.z = 0; 
                cloud_flat->points.push_back(p_flat);
            }

            pcl::SACSegmentation<pcl::PointXYZI> seg;
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_LINE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.15); 
            seg.setMaxIterations(200);      
            seg.setInputCloud(cloud_flat);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() < 8) return res;

            float dx = coefficients->values[3];
            float dy = coefficients->values[4];
            float px = coefficients->values[0];
            float py = coefficients->values[1];

            
            res.valid = true;
            res.angle = std::atan2(dy, dx);
            res.inliers = inliers->indices.size();
            float A = -dy; float B = dx; float C = dy*px - dx*py;
            res.distance = std::abs(C) / std::sqrt(A*A + B*B);
            return res;
        }
    
        pcl::PointCloud<pcl::PointXYZI>::Ptr filterByLinearity(const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_cloud) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZI>());
            if (in_cloud->empty()) return out_cloud;
    
            pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
            kdtree.setInputCloud(in_cloud);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
    
            for (size_t i = 0; i < in_cloud->points.size(); ++i) {
                if (kdtree.radiusSearch(in_cloud->points[i], config.geometry_check_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) >= 3) {
                    pcl::PCA<pcl::PointXYZI> pca;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr neighbors(new pcl::PointCloud<pcl::PointXYZI>);
                    for (int idx : pointIdxRadiusSearch) neighbors->push_back(in_cloud->points[idx]);
                    pca.setInputCloud(neighbors);
                    Eigen::Vector3f eigen_values = pca.getEigenValues();
                    float l1 = eigen_values(0); float l2 = eigen_values(1);
                    if (l1 > 1e-5) {
                        float linearity = (l1 - l2) / l1;
                        if (linearity > config.linearity_thresh) out_cloud->push_back(in_cloud->points[i]);
                    }
                }
            }
            return out_cloud;
        }
    };

class VanishingPointDetector {
    private:
        double smoothed_yaw_;       
        bool is_initialized_;
        bool has_lock_;
        int lock_frames_;
        cv::KalmanFilter KF_;
        cv::Mat measurement_;
        bool kf_initialized_;
        struct LineObj { 
            cv::Vec4f l; 
            float len; 
            float weight; 
        };
    
    public:
        VanishingPointDetector() : smoothed_yaw_(0.0), is_initialized_(false), has_lock_(false), lock_frames_(0), kf_initialized_(false) {
            KF_.init(4, 2, 0);
            KF_.transitionMatrix = (cv::Mat_<float>(4, 4) << 
                1, 0, 1, 0, 
                0, 1, 0, 1, 
                0, 0, 1, 0, 
                0, 0, 0, 1);
            setIdentity(KF_.measurementMatrix);
            setIdentity(KF_.processNoiseCov, cv::Scalar::all(1e-6));
            setIdentity(KF_.measurementNoiseCov, cv::Scalar::all(0.5));
            setIdentity(KF_.errorCovPost, cv::Scalar::all(1));
            measurement_ = cv::Mat::zeros(2, 1, CV_32F);
        }
        
        struct VPResult { 
            bool valid; 
            double yaw; 
            cv::Point2f pt; 
            int inliers; 
            std::vector<cv::Point> floor_poly; 
        }; 
    
        // [修改] 增强版 VP 检测：增加静止保护、方差检查和自动重置
    VPResult detect(cv::Mat& img_for_debug, bool is_robot_stationary) { 
        VPResult res = {false, 0.0, cv::Point2f(0,0), 0, {}};
        
        // 1. 静止保护 (Stationary Protection)
        // 如果机器人静止且已经锁定，我们大幅降低过程噪声，相当于“冻结”滤波器
        // 这样即使画面有噪点，VP 点也不会乱跑
        if (is_robot_stationary && has_lock_) {
             KF_.processNoiseCov = cv::Scalar::all(1e-7); // 冻结状态
        } else {
             KF_.processNoiseCov = cv::Scalar::all(1e-4); // 正常更新
        }

        cv::Mat gray;
        if (img_for_debug.channels() == 3) cv::cvtColor(img_for_debug, gray, cv::COLOR_BGR2GRAY);
        else gray = img_for_debug.clone();
        
        // 预处理增强：在暗处增加对比度
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(gray, gray);

        std::vector<cv::Vec4i> lines_i;
        cv::Canny(gray, gray, 50, 200, 3); 
        cv::HoughLinesP(gray, lines_i, 1, CV_PI/180, 50, 60, 10);
        
        std::vector<LineObj> candidates;
        
        // [新增] 严格的角度过滤器
        for(auto l_raw : lines_i) {
            cv::Vec4f l(l_raw[0], l_raw[1], l_raw[2], l_raw[3]);
            float dx = l[2] - l[0]; 
            float dy = l[3] - l[1];
            float len = std::sqrt(dx*dx + dy*dy);
            if (len < 40.0) continue; 
            
            float angle_deg = std::atan2(dy, dx) * 180.0 / CV_PI;
            
            // 过滤掉水平线（走廊线条通常是纵向或斜向汇聚的）
            // 如果全是水平线（如百叶窗），容易导致 VP 漂移到无穷远
            if (std::abs(angle_deg) < 20 || std::abs(angle_deg) > 85) continue;
            
            candidates.push_back({l, len, std::pow(len, 1.2f)});
        }

        if (candidates.size() < 3) { 
            has_lock_ = false; 
            // 如果线太少，直接重置 KF 初始化状态，防止下一次检测被旧状态误导
            if (!has_lock_) kf_initialized_ = false; 
            return res; 
        }
        
        // RANSAC 投票逻辑 (保持原有逻辑，稍作参数调整)
        int w = gray.cols; int h = gray.rows;
        double best_total_score = 0.0; 
        int best_inlier_count = 0; 
        cv::Point2f raw_best_vp(w/2, h/2);
        int iter_num = std::min(400, (int)candidates.size() * 15);
        
        for (int i = 0; i < iter_num; ++i) { 
            int idx1 = rand() % candidates.size(); 
            int idx2 = rand() % candidates.size();
            if (idx1 == idx2) continue;
            cv::Point2f vp = getIntersection(candidates[idx1].l, candidates[idx2].l);
            if (vp.x < -w || vp.x > 2*w) continue;
            if (vp.y < h * 0.20 || vp.y > h * 0.70) continue; // 限制在地平线附近

            double current_weight = 0.0; 
            int current_count = 0;
            for (const auto& obj : candidates) {
                if (distPointToLine(vp, obj.l) < 10.0) { 
                    current_weight += obj.weight; 
                    current_count++; 
                }
            }
            
            // 惩罚偏离图像中心过远的点（除非正在转向）
            double dist_from_center_x = std::abs(vp.x - w/2.0);
            double center_penalty = 1.0;
            if (dist_from_center_x > w * 0.3) 
                center_penalty = std::max(0.6, 1.0 - (dist_from_center_x - w*0.3)/(w*0.5));
            
            double final_score = current_weight * center_penalty;
            if (final_score > best_total_score) { 
                best_total_score = final_score; 
                best_inlier_count = current_count; 
                raw_best_vp = vp; 
            }
        }

        // [新增] 动态阈值：非锁定状态下需要更多内点才能激活，防止误触发
        int required_inliers = has_lock_ ? 5 : 8; 

        if (best_inlier_count > required_inliers) { 
            cv::Mat prediction = KF_.predict();
            cv::Point2f predicted_pt(prediction.at<float>(0), prediction.at<float>(1));
            
            if (!kf_initialized_ || !has_lock_) {
                // 初始化 KF
                KF_.statePre.at<float>(0) = raw_best_vp.x;
                KF_.statePre.at<float>(1) = raw_best_vp.y;
                KF_.statePre.at<float>(2) = 0;
                KF_.statePre.at<float>(3) = 0;
                KF_.statePost = KF_.statePre.clone(); 
                predicted_pt = raw_best_vp;
                kf_initialized_ = true;
                lock_frames_ = 0;
            } else {
                // [关键修改] 发散检测 (Innovation Check)
                // 如果测量值和预测值偏差极大（比如 > 150像素），说明可能检测到了错误的墙缝或噪点
                // 此时不要强行平滑，而是选择：1. 忽略该帧（保持原样） 或 2. 如果长期偏差则重置
                double dist_pred = cv::norm(predicted_pt - raw_best_vp);
                
                if (dist_pred > 150.0) {
                    if (lock_frames_ > 30) {
                        // 之前很稳定，突然跳变 -> 认为是噪点，使用预测值代替测量值
                        raw_best_vp = predicted_pt; 
                    } else {
                        // 之前不稳定，突然跳变 -> 可能是场景变了，强制重置 KF 到新位置
                        KF_.statePost.at<float>(0) = raw_best_vp.x;
                        KF_.statePost.at<float>(1) = raw_best_vp.y;
                        KF_.statePost.at<float>(2) = 0;
                        KF_.statePost.at<float>(3) = 0;
                    }
                } else {
                    // 正常更新
                    measurement_.at<float>(0) = raw_best_vp.x;
                    measurement_.at<float>(1) = raw_best_vp.y;
                    KF_.correct(measurement_);
                    raw_best_vp.x = KF_.statePost.at<float>(0);
                    raw_best_vp.y = KF_.statePost.at<float>(1);
                }
            }

            // 最后的 NaN 检查，防止程序崩溃
            if (!std::isfinite(raw_best_vp.x) || !std::isfinite(raw_best_vp.y)) {
                res.valid = false; has_lock_ = false; kf_initialized_ = false;
                return res;
            }

            res.valid = true; 
            res.pt = raw_best_vp; 
            res.inliers = best_inlier_count;
            has_lock_ = true; 
            if (lock_frames_ < 100) lock_frames_++;
            
            double measured_yaw = -std::atan((res.pt.x - config.cam_cx) / config.cam_fx);
            smoothed_yaw_ = measured_yaw; 
            res.yaw = smoothed_yaw_;
            
            // Debug 画图
            for (const auto& obj : candidates) {
                if (distPointToLine(res.pt, obj.l) < 10.0) 
                    cv::line(img_for_debug, cv::Point(obj.l[0], obj.l[1]), cv::Point(obj.l[2], obj.l[3]), cv::Scalar(0, 255, 0), 2);
            }
            generateFloorPolygon(res, w, h, candidates, gray);
        } else { 
            // 丢失锁定
            has_lock_ = false; 
            lock_frames_ = 0;
            // [建议] 丢失后重置初始化状态，避免下次重连时被旧的漂移值影响
            kf_initialized_ = false; 
        }
        return res;
    }
    
    private:
        void generateFloorPolygon(VPResult& res, int w, int h, const std::vector<LineObj>& lines, const cv::Mat& gray) {
            cv::Point2f vp = res.pt;
            int roi_width_px = w * config.roi_w_ratio;
            float limit_left = 5.0 + roi_width_px;     
            float limit_right = w - 5.0 - roi_width_px;
            cv::Point2f bottom_left(limit_left, h); 
            cv::Point2f bottom_right(limit_right, h);
            res.floor_poly.push_back(cv::Point(vp.x, vp.y));
            res.floor_poly.push_back(bottom_right);
            res.floor_poly.push_back(bottom_left);
        }
    
        cv::Point2f getIntersection(cv::Vec4f l1, cv::Vec4f l2) {
            float x1=l1[0], y1=l1[1], x2=l1[2], y2=l1[3]; 
            float x3=l2[0], y3=l2[1], x4=l2[2], y4=l2[3];
            float d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
            if (std::abs(d) < 1e-3) return cv::Point2f(-1.0f, -1.0f); 
            return cv::Point2f(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/d, ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/d);
        }
    
        float distPointToLine(cv::Point2f p, cv::Vec4f l) {
            float A = p.x - l[0]; float B = p.y - l[1]; float C = l[2] - l[0]; float D = l[3] - l[1];
            float len_sq = C * C + D * D; if (len_sq == 0) return -1;
            return std::abs(C * (l[1] - p.y) - (l[0] - p.x) * D) / std::sqrt(len_sq);
        }
    };

class VisualFlowEstimator {
    private:
        cv::Mat prev_gray_;
        cv::Mat prev_depth_; 
        std::vector<cv::Point2f> prev_pts_;
        bool is_initialized_;
        
    public:
        VisualFlowEstimator() : is_initialized_(false) {}
    
        std::tuple<gtsam::Vector3, double, double> estimate(const cv::Mat& curr_gray, const cv::Mat& curr_depth, double dt) {
            if (dt < 0.01) return {gtsam::Vector3::Zero(), 0.0, 0.0};
            
            if (!is_initialized_ || prev_pts_.size() < config.flow_min_pts) {
                std::vector<cv::Point2f> new_pts;
                int grid_rows = 3;
                int grid_cols = 4;
                int cell_w = curr_gray.cols / grid_cols;
                int cell_h = curr_gray.rows / grid_rows;
                int max_pts_per_cell = 400 / (grid_rows * grid_cols);

                for (int r = 0; r < grid_rows; r++) {
                    for (int c = 0; c < grid_cols; c++) {
                        cv::Rect cell_roi(c * cell_w, r * cell_h, cell_w, cell_h);
                        if (cell_roi.x + cell_roi.width > curr_gray.cols) cell_roi.width = curr_gray.cols - cell_roi.x;
                        if (cell_roi.y + cell_roi.height > curr_gray.rows) cell_roi.height = curr_gray.rows - cell_roi.y;
                        std::vector<cv::Point2f> cell_pts;
                        cv::goodFeaturesToTrack(curr_gray(cell_roi), cell_pts, max_pts_per_cell, 0.001, 3.0);
                        for (auto& p : cell_pts) {
                            p.x += cell_roi.x;
                            p.y += cell_roi.y;
                            new_pts.push_back(p);
                        }
                    }
                }
                
                if (!new_pts.empty()) {
                    if (is_initialized_ && prev_pts_.size() > 0) {
                        prev_pts_.insert(prev_pts_.end(), new_pts.begin(), new_pts.end());
                    } else {
                        prev_pts_ = new_pts;
                    }
                    prev_gray_ = curr_gray.clone();
                    prev_depth_ = curr_depth.clone();
                    is_initialized_ = true;
                    return {gtsam::Vector3::Zero(), 0.0, 0.0};
                }
            }
    
            if (prev_pts_.empty()) {
                return {gtsam::Vector3::Zero(), 0.0, 0.0};
            }
    
            std::vector<cv::Point2f> curr_pts;
            std::vector<uchar> status;
            std::vector<float> err;
            cv::Size winSize(31, 31); 
            int maxLevel = 4;

            cv::calcOpticalFlowPyrLK(
                prev_gray_,      
                curr_gray,       
                prev_pts_,       
                curr_pts,        
                status,         
                err,             
                winSize,         
                maxLevel,        
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                0,               
                0.001            
            );

            std::vector<cv::Point2f> rev_pts;    
            std::vector<uchar> rev_status;       
            std::vector<float> rev_err;          

            cv::calcOpticalFlowPyrLK(
                curr_gray,
                prev_gray_,     
                curr_pts,       
                rev_pts,        
                rev_status,
                rev_err,
                winSize,
                maxLevel,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                0,
                0.001
            );
    
            std::vector<gtsam::Vector3> valid_velocities;
            std::vector<double> valid_depths; 
            std::vector<cv::Point2f> good_curr_pts;
            int rejected_by_fb = 0; 

            for (size_t i = 0; i < prev_pts_.size(); ++i) {
                if (!status[i]) continue;
                if (!rev_status[i] || cv::norm(prev_pts_[i] - rev_pts[i]) > 1.0) {
                    rejected_by_fb++;
                    continue; 
                }
                int u_prev = std::round(prev_pts_[i].x); int v_prev = std::round(prev_pts_[i].y);
                int u_curr = std::round(curr_pts[i].x); int v_curr = std::round(curr_pts[i].y);
                if (u_prev < 0 || u_prev >= curr_depth.cols || v_prev < 0 || v_prev >= curr_depth.rows) continue;
                if (u_curr < 0 || u_curr >= curr_depth.cols || v_curr < 0 || v_curr >= curr_depth.rows) continue;
                unsigned short d_prev_raw = prev_depth_.at<unsigned short>(v_prev, u_prev);
                unsigned short d_curr_raw = curr_depth.at<unsigned short>(v_curr, u_curr);
                if (d_prev_raw == 0 || d_curr_raw == 0) continue;
                float z_prev = d_prev_raw * 0.001f; float z_curr = d_curr_raw * 0.001f;
                if (z_prev < 0.3 || z_prev > 8.0 || z_curr < 0.3 || z_curr > 8.0) continue;
                float x_prev = (u_prev - config.cam_cx) * z_prev / config.cam_fx;
                float y_prev = (v_prev - config.cam_cy) * z_prev / config.cam_fy;
                float x_curr = (u_curr - config.cam_cx) * z_curr / config.cam_fx;
                float y_curr = (v_curr - config.cam_cy) * z_curr / config.cam_fy;
                gtsam::Point3 p_prev(x_prev, y_prev, z_prev);
                gtsam::Point3 p_curr(x_curr, y_curr, z_curr);
                gtsam::Vector3 v_cam = (p_prev - p_curr) / dt; 
                gtsam::Vector3 v_body(v_cam.z(), -v_cam.x(), -v_cam.y());
                valid_velocities.push_back(v_body);
                valid_depths.push_back((z_prev + z_curr) / 2.0); 
                good_curr_pts.push_back(curr_pts[i]);
            }
            if (valid_velocities.size() < 10) {
                prev_gray_ = curr_gray.clone();
                prev_depth_ = curr_depth.clone();
                prev_pts_ = good_curr_pts; 
                return {gtsam::Vector3::Zero(), 0.0, 0.0}; 
            }
            gtsam::Vector3 best_vel(0,0,0);
            double best_avg_depth = 0.0; 
            int max_inliers = 0;
            
            for (int i=0; i< std::min((int)valid_velocities.size(), 20); ++i) { 
                int inliers = 0;
                gtsam::Vector3 seed_vel = valid_velocities[i];
                gtsam::Vector3 sum_weighted_vel(0,0,0);
                double sum_weights = 0.0;
                double sum_depths = 0.0;
                for (size_t k = 0; k < valid_velocities.size(); ++k) {
                    if ((valid_velocities[k] - seed_vel).norm() < config.flow_ransac_thresh) {
                        inliers++;
                        double z = valid_depths[k];
                        double weight = 1.0 / (z * z + 1e-5);
                        sum_weighted_vel += valid_velocities[k] * weight;
                        sum_weights += weight;
                        sum_depths += z;
                    }
                }
                if (inliers > max_inliers) {
                    max_inliers = inliers;
                    if (sum_weights > 1e-6) {
                        best_vel = sum_weighted_vel / sum_weights; 
                        best_avg_depth = sum_depths / inliers;     
                    }
                }
            }
            prev_gray_ = curr_gray.clone();
            prev_depth_ = curr_depth.clone();
            prev_pts_ = good_curr_pts;
            if (max_inliers > 5) {
                double confidence = (double)max_inliers / (double)valid_velocities.size();
                confidence = std::min(1.0, std::max(0.0, confidence));
                double vel_norm = best_vel.norm();
                if (vel_norm > 1.0) {
                    is_initialized_ = false;
                    prev_pts_.clear();
                    return {gtsam::Vector3::Zero(), 0.0, 0.0};
                }
                if (confidence < 0.15) {
                     is_initialized_ = false;
                     prev_pts_.clear();
                     return {gtsam::Vector3::Zero(), 0.0, 0.0};
                }
                return {best_vel, confidence, best_avg_depth}; 
            } else {
                is_initialized_ = false; 
                prev_pts_.clear();
                return {gtsam::Vector3::Zero(), 0.0, 0.0};
            }
        }
    };

class FactorGraphTracker {
public:
    gtsam::Vector2 current_calib_;

    FactorGraphTracker() : key_index_(0), last_keyframe_time_(0) {
        current_pose_ = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0,0,0));
        current_velocity_ = gtsam::Vector3(0,0,0);
        current_calib_ = gtsam::Vector2(1.0, 1.0); 
        K_ = boost::make_shared<gtsam::Cal3_S2>(config.cam_fx, config.cam_fy, 0.0, config.cam_cx, config.cam_cy);
        last_keyframe_pose_ = current_pose_; 
        hardReset(current_pose_, current_velocity_);
    }


    // 在 class FactorGraphTracker 的 public: 下面添加
    void hardReset(const gtsam::Pose3& pose, const gtsam::Vector3& vel) {
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.1;
        params.relinearizeSkip = 1;
        isam_ = std::make_unique<gtsam::ISAM2>(params);

        current_pose_ = pose;
        current_velocity_ = vel;
        current_calib_ = gtsam::Vector2(1.0, 1.0); 

        last_keyframe_pose_ = current_pose_;
        last_keyframe_time_ = ros::Time(0); 

        gtsam::NonlinearFactorGraph new_graph;
        gtsam::Values new_values;

        // 强先验，强制锁定位置
        new_graph.add(PriorFactor<Pose3>(X(key_index_), current_pose_, 
            noiseModel::Diagonal::Sigmas((Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished())));
        new_graph.add(PriorFactor<Vector3>(V(key_index_), current_velocity_, 
            noiseModel::Isotropic::Sigma(3, 0.1)));
        new_graph.add(PriorFactor<Vector2>(C(key_index_), current_calib_, 
            noiseModel::Diagonal::Sigmas(Vector2(0.05, 0.05)))); 

        new_values.insert(X(key_index_), current_pose_);
        new_values.insert(V(key_index_), current_velocity_);
        new_values.insert(C(key_index_), current_calib_);

        isam_->update(new_graph, new_values);
        
        
    }
        // [修复编译报错版] predictAndUpdate
    // [完整修复版] predictAndUpdate
    // 包含：摆渡模式屏蔽、柔性网格吸附、防炸图轨道锁定、智能航向仲裁
    void predictAndUpdate(ros::Time now, bool leg_trigger_active, 
        double estimated_spacing, double periodicity_conf, 
        bool has_vp, double vp_yaw, cv::Point2f vp_pixel, double vp_conf, 
        const gtsam::Rot3& calib_rot_correction,
        double wheel_vel_x, double wheel_omega,
        bool has_plane, const gtsam::Vector4& plane_coeffs,
        bool wall_detected,        
        bool front_wall_detected,  
        bool has_vis_vel, const gtsam::Vector3& vis_vel, double vis_conf, double avg_vis_depth,
        bool has_line, const StructuralFeatureExtractor::LineResult& line_res,
        bool use_course_lock, double target_course_yaw,
        gtsam::Point3 anchor_point,
        bool is_ferrying,
        double lio_lateral_vel) {  // <--- [🔴 新增参数]
        
        double log_structure_sigma = 0.0; // [新增] 用于记录本帧约束强度

        // --- 关键帧检查 ---
        if (last_keyframe_time_.isZero()) {
            last_keyframe_time_ = now;
            last_keyframe_pose_ = current_pose_; 
            return;
        }

        double dt = (now - last_keyframe_time_).toSec();
        double dist_moved = (current_pose_.translation() - last_keyframe_pose_.translation()).norm();
        double angle_moved = gtsam::Rot3::Logmap(current_pose_.rotation().between(last_keyframe_pose_.rotation())).norm();

        bool need_keyframe = false;
        if (dist_moved > 0.05 || angle_moved > 0.035) need_keyframe = true;
        if (dt > 0.5) need_keyframe = true;
        if (leg_trigger_active && !is_ferrying) need_keyframe = true; // 摆渡时不因触发器强制关键帧

        if (!need_keyframe) return;

        if (dt > 0.5) {
            ROS_INFO_THROTTLE(2.0, "[Keyframe] Reason: TIME | dt: %.3f", dt);
        } else {
            ROS_INFO_THROTTLE(2.0, "[Keyframe] Reason: MOVE | Dist: %.3f", dist_moved);
        }

        key_index_++;
        last_keyframe_time_ = now;
        last_keyframe_pose_ = current_pose_; 

        gtsam::NonlinearFactorGraph new_factors;
        gtsam::Values new_values;

        bool is_stationary = std::abs(wheel_vel_x) < 0.02 && std::abs(wheel_omega) < 0.02;
        bool is_reversing = (wheel_vel_x < -0.05); 
        bool is_startup_phase = (key_index_ < 60);
        
        // 曼哈顿约束开关 (在函数内定义，方便后续逻辑使用)
        bool use_manhattan = !has_vp; // 默认策略：没有 VP 时才尝试用曼哈顿

        gtsam::Pose3 pred_pose;
        gtsam::Vector3 pred_vel;

        // =========================================================
        // --- 1. 运动学预测 (Odometry) & 混合速度约束 (LIO Lateral) ---
        // =========================================================
        if (is_stationary) {
            pred_pose = current_pose_;
            pred_vel = gtsam::Vector3(0,0,0);
            
            // 静止状态下的强约束：位置不变，速度为零，标定参数不变
            new_factors.add(BetweenFactor<Pose3>(X(key_index_-1), X(key_index_), Pose3(Rot3(), Point3(0,0,0)), 
                noiseModel::Diagonal::Sigmas((Vector(6) << 1e-4, 1e-4, 1e-3, 5e-3, 5e-3, 5e-3).finished())));
            
            new_factors.add(PriorFactor<Vector3>(V(key_index_), gtsam::Vector3(0,0,0), noiseModel::Isotropic::Sigma(3, 1e-5)));
            
            new_factors.add(BetweenFactor<Vector2>(C(key_index_-1), C(key_index_), gtsam::Vector2(0.0, 0.0), 
                noiseModel::Diagonal::Sigmas(gtsam::Vector2(1e-6, 1e-6))));
        } else {
            double pred_v = wheel_vel_x * current_calib_(0);
            double pred_w = wheel_omega * current_calib_(1); 
            
            // [🔴 修改] 预测步骤：加入 LIO 的横向速度分量
            // 之前的逻辑假设 dy=0，现在使用 LIO 估计的侧滑速度进行位姿推演
            // 这确保了预测轨迹本身就包含了侧滑，减少了与视觉观测的冲突
            gtsam::Pose3 odom_delta = gtsam::Pose3(
                gtsam::Rot3::Ypr(pred_w * dt, 0, 0), 
                gtsam::Point3(pred_v * dt, lio_lateral_vel * dt, 0) // <--- 关键点：注入横向位移
            );
            
            pred_pose = current_pose_.compose(odom_delta);
            pred_vel = gtsam::Vector3(pred_v, lio_lateral_vel, 0); // <--- 更新预测速度

            // TrackKinematicsFactor (相对约束 - 保持原有逻辑)
            double abs_w = std::abs(wheel_omega);
            gtsam::Vector6 odom_sigmas;
            
            if (is_reversing) {
                // 倒车时放宽约束
                odom_sigmas << 0.15, 0.15, 0.2, 0.3, 0.2, 0.2;  
            } else {
                // 正常行驶
                odom_sigmas << 0.1, 0.1, 0.1, 0.2, 0.1, 0.1;
            }

            new_factors.add(boost::make_shared<TrackKinematicsFactor>(
                X(key_index_-1), X(key_index_), C(key_index_-1),
                wheel_vel_x, wheel_omega, dt, 
                noiseModel::Robust::Create(
                    noiseModel::mEstimator::Huber::Create(1.345), 
                    noiseModel::Diagonal::Sigmas(odom_sigmas)
                )
            ));

            // 自适应滑移参数学习 (保持原有逻辑)
            double current_slip_sigma;
            if (is_reversing) {
                current_slip_sigma = 1e-6; 
            } else {
                double maneuver_stress = abs_w;
                if (std::abs(current_velocity_.x()) > 0.05 && std::abs(wheel_vel_x) > 0.05 && current_velocity_.x() * wheel_vel_x < 0) {
                    maneuver_stress = 5.0; 
                }
                current_slip_sigma = (config.slip_learn_sigma_base * 0.05) + (config.slip_learn_sigma_gain * maneuver_stress * 0.1);
                current_slip_sigma = std::min(0.01, current_slip_sigma);
            }

            new_factors.add(BetweenFactor<Vector2>(
                C(key_index_-1), C(key_index_), gtsam::Vector2(0.0, 0.0), 
                noiseModel::Diagonal::Sigmas(gtsam::Vector2(current_slip_sigma, current_slip_sigma * 2.0))));
            
            new_factors.add(PriorFactor<Vector2>(C(key_index_), gtsam::Vector2(1.18, 1.0), 
                noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.1, 0.2)))); 

            // [🔴 核心修改] 混合速度约束 (Hybrid Velocity Constraint)
            // 目标：Vx 使用编码器(纵向准确)，Vy 使用 LIO(横向侧滑准确)
            
            double target_vx = wheel_vel_x * current_calib_(0);
            double target_vy = lio_lateral_vel; 

            // 动态协方差设计：
            // Vx Sigma: 0.5 (信任编码器，但允许滑动)
            // Vy Sigma: 0.05 (非常信任 LIO 的侧向检测，用于纠正漂移，防止波浪状轨迹)
            // Vz Sigma: 0.01 (锁死)
            gtsam::Vector3 vel_sigmas(0.5, 0.05, 0.01);
            
            if (is_reversing) vel_sigmas(0) = 1.0; // 倒车时 Vx 不准，进一步放宽

            new_factors.add(PriorFactor<Vector3>(V(key_index_), 
                gtsam::Vector3(target_vx, target_vy, 0.0), 
                noiseModel::Diagonal::Sigmas(vel_sigmas)
            ));

            // 速度平滑约束
            new_factors.add(BetweenFactor<Vector3>(V(key_index_-1), V(key_index_), gtsam::Vector3(0,0,0), noiseModel::Isotropic::Sigma(3, 0.5))); 
        }

        current_pose_ = pred_pose;
        current_velocity_ = pred_vel;
        new_values.insert(X(key_index_), current_pose_);
        new_values.insert(V(key_index_), current_velocity_);
        new_values.insert(C(key_index_), current_calib_);


        // =========================================================
        // --- [🔴 摆渡模式判断] ---
        // 如果处于摆渡模式 (is_ferrying = true)，跳过后续大部分几何约束
        // =========================================================
        
        if (!is_ferrying) {

            // --- 2. 视觉灭点 (VP) ---
            // 增加角速度限制：只有在走直线时才信任灭点，转弯时(omega > 0.2)不信任
            // --- 2. 视觉灭点 (VP) ---
            // [修改 2] 智能阈值：如果是轨道锁定状态(use_course_lock)，放宽转弯判定
            // 倒车时摆动大，阈值设为 0.4；普通模式设为 0.2
            double turn_thresh = use_course_lock ? 0.4 : 0.2;

            if (std::abs(wheel_omega) < turn_thresh) {
                new_factors.add(boost::make_shared<VanishingPointFactor>(X(key_index_), 0.0,
                    noiseModel::Diagonal::Sigmas(Vector1(is_startup_phase ? 1e-4 : 0.3))));

                if (has_vp && !is_stationary) {
                    bool range_check_x = (vp_pixel.x > -2000.0 && vp_pixel.x < 3000.0);
                    bool range_check_y = (vp_pixel.y > -2000.0 && vp_pixel.y < 3000.0);
                    bool yaw_check = (std::abs(vp_yaw) < 1.0);

                    if (range_check_x && range_check_y && yaw_check) { 
                        gtsam::Rot3 R_b_c_base(0, 0, 1, -1, 0, 0, 0, -1, 0);
                        gtsam::Rot3 R_b_c_final = R_b_c_base.compose(calib_rot_correction);
                        gtsam::Pose3 T_b_c(R_b_c_final, gtsam::Point3(0.1, 0.0, config.camera_z_shift)); 

                        double sigma_pixel = 2.0 / (vp_conf + 1e-4); 
                        if (is_startup_phase) sigma_pixel *= 3.0;
                        
                        auto robust_model = noiseModel::Robust::Create(
                            noiseModel::mEstimator::Huber::Create(1.0), 
                            noiseModel::Isotropic::Sigma(2, sigma_pixel)
                        );

                        new_factors.add(boost::make_shared<ManifoldVPFactor>(
                            X(key_index_), vp_pixel, *K_, T_b_c, robust_model
                        ));
                    }
                }
            } else {
                ROS_INFO_THROTTLE(1.0, "[Turn] Turning (w=%.2f). VP Factors DISABLED.", wheel_omega);
            }

            // --- 3. 光流 (Flow) ---
            // 摆渡模式下也跳过光流，防止在大厅因动态物体产生误导
            if (has_vis_vel && !is_stationary && !front_wall_detected && !is_reversing) {
                double flow_sigma = 0.03 + 0.03 * (1.0 - vis_conf);
                double depth_penalty = 1.0 + 0.05 * (avg_vis_depth * avg_vis_depth);
                flow_sigma *= depth_penalty;

                // ================= [🔴 补全 1: 写入光流日志] =================
                if (g_logger.f_flow_stats.is_open()) {
                    g_logger.f_flow_stats << std::fixed << std::setprecision(5)
                        << now.toSec() << " "
                        << vis_vel.x() << " "     // 视觉测速
                        << pred_vel.x() << " "    // 预测速度
                        << vis_conf << " "        // 置信度
                        << avg_vis_depth << " "   // 平均深度
                        << flow_sigma << " "      // 计算出的 Sigma
                        << 1.0                    // Active 标记
                        << std::endl;
                }
                // ==========================================================

                gtsam::Vector3 flow_sigmas(flow_sigma, flow_sigma * 1.5, flow_sigma * 1.5);
                gtsam::Vector3 constrained_vis_vel = vis_vel;
                constrained_vis_vel(1) = 0.0; 
                
                gtsam::Vector3 error = constrained_vis_vel - pred_vel;
                double mahalanobis_dist = 0.0;
                for(int i=0; i<3; ++i) mahalanobis_dist += std::pow(error(i)/flow_sigmas(i), 2);
                mahalanobis_dist = std::sqrt(mahalanobis_dist);

                if (mahalanobis_dist < 15.0) {
                    new_factors.add(PriorFactor<Vector3>(V(key_index_), constrained_vis_vel, 
                        noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), noiseModel::Diagonal::Sigmas(flow_sigmas))));
                }
            }

            // --- 4. 线特征 (墙体几何约束) ---
            if (has_line) {
                double final_angle = line_res.angle; 
                double measured_dist = line_res.distance;

                // [新增] 墙体位置初始化与更新 (低通滤波)
                // 计算当前测量对应的“墙在哪里”： Current_Wall_Y = Pose_Y +/- Measured_Dist
                // 我们假设墙在机器人的一侧。为了简化，我们取绝对值距离。
                // 这里的逻辑假设：墙是平行于 X 轴的。
                
                double current_implied_wall_y = 0.0;
                
                // 简单的启发式判断：墙在左边还是右边？
                // 如果 line_res 来自 pcl::SACSegmentation，它通常返回的是直线方程。
                // 这里我们假设 measure_dist 是正值，我们需要判断方向。
                // 这是一个简化策略：假设我们总是锁定离机器人最近的那堵墙，并假设它不动。
                
                double side_sign = (current_pose_.y() > 0) ? 1.0 : -1.0; // 假设 Y=0 是走廊中心
                // 如果你的坐标系不是以走廊为中心，这里可能需要调整
                // 更好的方式：利用上一次的 wall_y 来判断
                
                if (!wall_y_initialized_) {
                    // 初始化：假设机器人当前位置是准确的，反推墙的位置
                    if (std::abs(current_pose_.y()) < 0.1) { // 只有在中心附近才初始化
                         // 这是一个猜测，假设墙在 +Y 或 -Y 方向
                         // 你可能需要根据实际点云的重心来判断墙在左还是右
                         estimated_wall_y_ = current_pose_.y() + measured_dist; // 默认 +Y
                         // 或者根据 line_res 的系数来判断方向
                    }
                    // 暂时跳过初始化帧
                } else {
                    // 判断测量的是哪一边的墙 (对比 estimated_wall_y_)
                    double check_pos = current_pose_.y() + measured_dist;
                    double check_neg = current_pose_.y() - measured_dist;
                    
                    if (std::abs(check_pos - estimated_wall_y_) < std::abs(check_neg - estimated_wall_y_)) {
                        current_implied_wall_y = check_pos;
                    } else {
                        current_implied_wall_y = check_neg;
                    }

                    // 平滑更新墙的位置 (这就像是一个 SLAM 的 Mapping 过程)
                    // Alpha = 0.05, 也就是我们非常信任历史的墙位置（因为墙不会动）
                    estimated_wall_y_ = 0.95 * estimated_wall_y_ + 0.05 * current_implied_wall_y;
                    
                    if (!wall_y_initialized_) { estimated_wall_y_ = current_implied_wall_y; wall_y_initialized_ = true; }

                    // [关键] 添加双重约束因子 (角度 + 距离)
                    // 动态 Sigma: 内点越多，权重越高 (Sigma越小)
                    double dynamic_sigma = std::max(0.02, std::min(0.5, 0.1 * (30.0 / (line_res.inliers + 1.0)))); 
                    
                    // 距离的权重应该比角度更敏感，因为我们想拉直墙面
                    // 这里我们构造一个 Vector2 的噪声模型
                    auto line_noise = noiseModel::Diagonal::Sigmas(gtsam::Vector2(dynamic_sigma, dynamic_sigma * 0.5));

                    new_factors.add(boost::make_shared<SemanticLineFactor>(
                        X(key_index_), final_angle, measured_dist, estimated_wall_y_,
                        noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), line_noise)
                    ));
                    
                    // Debug Log
                    // if (key_index_ % 10 == 0) std::cout << "Wall Locked at Y=" << estimated_wall_y_ << " Dist=" << measured_dist << std::endl;
                }
            }

            // --- 5. 周期性结构 (网格吸附 - 安全熔断版) ---
            if (leg_trigger_active && !front_wall_detected) { 
                double norm_conf = std::min(1.0, periodicity_conf); 
                double current_x = pred_pose.x();
                double target_x = std::round(current_x / estimated_spacing) * estimated_spacing;
                double snap_error = std::abs(current_x - target_x);

                // [⚡核心修改⚡] 熔断机制：如果误差超过间距的 25% (例如 0.4m)，说明吸附错了，直接放弃！
                // 强行吸附会导致因子图崩溃 (GTSAM Error)
                if (snap_error > (estimated_spacing * 0.25)) {
                    ROS_WARN_THROTTLE(1.0, "[Grid] SNAP REJECTED! Error too large: %.3fm (Limit: %.3fm)", 
                        snap_error, estimated_spacing * 0.25);
                    log_structure_sigma = 0.0; // 记录为0表示未添加
                } 
                else {
                    // 只有误差在可控范围内，才添加约束
                    double base_sigma = is_reversing ? 0.08 : 0.05; 
                    // 误差越大给的权重越低 (Sigma越大)
                    double dynamic_sigma = base_sigma + (snap_error * 2.0); 
                    double final_sigma = dynamic_sigma + 0.3 * (1.0 - norm_conf);
                    final_sigma = std::min(0.5, final_sigma);
                    
                    log_structure_sigma = final_sigma;

                    new_factors.add(boost::make_shared<StructurePeriodicityFactor>(
                        X(key_index_), estimated_spacing, 
                        noiseModel::Robust::Create(
                            noiseModel::mEstimator::Huber::Create(1.345), // <--- 关键保护
                            noiseModel::Diagonal::Sigmas(Vector1(final_sigma))
                        )
                    ));
                    ROS_INFO_THROTTLE(1.0, "[Grid Snap] Added. Err: %.3fm | Sigma: %.3f", snap_error, final_sigma);
                }
            }

            // --- 6. 地面约束 (走廊模式) ---
            if (has_plane) {
                new_factors.add(boost::make_shared<GroundPlaneFactor>(X(key_index_), plane_coeffs, noiseModel::Diagonal::Sigmas((Vector(4) << 0.1, 0.1, 0.1, 0.1).finished())));
            } else {
                new_factors.add(boost::make_shared<PlanarConstraintFactor>(X(key_index_), 0.0, noiseModel::Diagonal::Sigmas(Vector3(0.05, 0.05, 0.05)))); 
            }

            // --- 7. 航向约束 & 轨道锁定 ---

            
            // --- 7. 航向约束 & 轨道锁定 ---
            
            // [修改 1] 移除 is_reversing 检查，只要 upstream 发送了锁定信号就执行
            if (use_course_lock) { 
                double current_yaw = pred_pose.rotation().yaw();
                double yaw_error = current_yaw - target_course_yaw;
                while(yaw_error > M_PI) yaw_error -= 2*M_PI;
                while(yaw_error < -M_PI) yaw_error += 2*M_PI;

                // 只有偏差不大的时候才强行修正，防止瞬移
                if (std::abs(yaw_error) < 0.50) { 
                    // 再次检查角速度，确保不是正在猛打方向盘
                    if (std::abs(wheel_omega) < 0.20) {
                        double curr_roll = pred_pose.rotation().roll();
                        double curr_pitch = pred_pose.rotation().pitch();
                        gtsam::Rot3 target_rot = gtsam::Rot3::Ypr(target_course_yaw, curr_pitch, curr_roll);
                        
                        gtsam::Pose3 target_pose(target_rot, anchor_point);
                        
                        // [修改 3] 动态轨道约束：前进紧，后退松
                        // 倒车时给 15cm (0.15) 的横向自由度，允许蛇形走位，但大方向必须对
                        double current_rail_y_sigma = is_reversing ? 0.15 : 0.005; 
                        double current_rail_yaw_sigma = is_reversing ? 0.10 : 0.02; // 倒车允许角度晃动

                        auto rail_noise = noiseModel::Diagonal::Sigmas(
                            (gtsam::Vector(6) << 
                             1.0, 1.0, current_rail_yaw_sigma, // Roll, Pitch, Yaw
                             2.0,                  // X (自由)
                             current_rail_y_sigma, // Y (动态调整)
                             0.05                  // Z
                            ).finished()
                        );
                        
                        // 始终添加约束 (使用 Huber 核函数保护防止炸图)
                        new_factors.add(PriorFactor<Pose3>(X(key_index_), target_pose, 
                            noiseModel::Robust::Create(
                                noiseModel::mEstimator::Huber::Create(1.345), 
                                rail_noise
                            )));
                        
                        if (is_reversing) {
                            ROS_INFO_THROTTLE(2.0, "[Rail Mode] Soft Lock Active (Reversing). Sigma Y: %.2f", current_rail_y_sigma);
                        }
                        
                        // 侧向零速约束 (Non-Holonomic Constraint)
                        // 履带车无法横移，强力约束侧向速度 Vy (1e-4)
                        new_factors.add(PriorFactor<Vector3>(V(key_index_), gtsam::Vector3(current_velocity_.x(), 0.0, 0.0), 
                           noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.5, 1e-6, 0.5))));
                           
                        ROS_INFO_THROTTLE(2.0, "[Rail Mode] ON. Locked Y: %.3f", anchor_point.y());
                    }
                } 
            }

        } else { 
            // =========================================================
            // --- [摆渡模式执行] (Ferrying Mode Active) ---
            // 此时只添加非常松散的 Z 轴约束防止飞天，其余完全依赖里程计
            // =========================================================
            
            // 摆渡时添加一个松散的平面约束 (Z=0)，防止长时间纯积分导致高度漂移
            new_factors.add(boost::make_shared<PlanarConstraintFactor>(X(key_index_), 0.0, 
                noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.5)))); 
                
            ROS_INFO_THROTTLE(2.0, "[Tracker] FERRYING MODE. Pure Odometry Active.");
        }

        // --- 8. 退化保护 ---
        if (is_degenerate_ && !has_vis_vel && !leg_trigger_active) {
            new_factors.add(BetweenFactor<Vector3>(V(key_index_-1), V(key_index_), gtsam::Vector3(0,0,0), 
                noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.1, 0.5, 0.5))));
        }

        // --- 9. GTSAM Update ---
        try {
            isam_->update(new_factors, new_values);
            isam_->update(); 

            current_pose_ = isam_->calculateEstimate<Pose3>(X(key_index_));
            current_velocity_ = isam_->calculateEstimate<Vector3>(V(key_index_));
            current_calib_ = isam_->calculateEstimate<Vector2>(C(key_index_)); 

            // 协方差与退化检测
            static int cov_calc_idx = 0;
            if (cov_calc_idx++ % 20 == 0) {
                try {
                    current_covariance_ = isam_->marginalCovariance(X(key_index_));
                    gtsam::Matrix3 pos_cov = current_covariance_.block<3, 3>(3, 3);
                    Eigen::SelfAdjointEigenSolver<gtsam::Matrix3> eigensolver(pos_cov);
                    if (eigensolver.info() == Eigen::Success) {
                        double max_eigenvalue = eigensolver.eigenvalues()(2); 
                        if (g_logger.f_eigen.is_open()) {
                            g_logger.f_eigen << std::fixed << std::setprecision(4) 
                                << now.toSec() << " " << max_eigenvalue << " " 
                                << (max_eigenvalue > 0.05 ? 1 : 0) << " " 
                                << (leg_trigger_active ? 1 : 0) << " " 
                                << log_structure_sigma  // [修改] 使用捕获的变量替换 0.0
                                << std::endl;
                        }
                        if (max_eigenvalue > 0.05) { 
                            is_degenerate_ = true;
                            ROS_WARN_THROTTLE(2.0, "[Degeneracy] High Uncertainty! MaxEig: %.4f", max_eigenvalue);
                        } else {
                            is_degenerate_ = false;
                        }
                    }
                } catch (gtsam::IndeterminantLinearSystemException& e) {
                    is_degenerate_ = true;
                }
            }

            // 数据记录 (DataLogger)
            if (g_logger.f_traj.is_open()) {
                g_logger.f_traj << std::fixed << std::setprecision(6) << now.toSec() << " "
                << current_pose_.x() << " " << current_pose_.y() << " " << current_pose_.z() << " "
                << current_pose_.rotation().toQuaternion().x() << " " << current_pose_.rotation().toQuaternion().y() << " "
                << current_pose_.rotation().toQuaternion().z() << " " << current_pose_.rotation().toQuaternion().w() << " "
                << current_velocity_.x() << " " << current_velocity_.y() << " " << current_velocity_.z() << std::endl;
            }

            if (g_logger.f_slip.is_open()) {
                double turn_rate = std::abs(wheel_omega);
                double maneuver_stress = turn_rate;
                if (std::abs(current_velocity_.x()) > 0.05 && std::abs(wheel_vel_x) > 0.05 && current_velocity_.x() * wheel_vel_x < 0) {
                    maneuver_stress = 5.0; 
                }
                double dynamic_sigma = config.slip_learn_sigma_base + config.slip_learn_sigma_gain * maneuver_stress;
                
                g_logger.f_slip << std::fixed << std::setprecision(6) << now.toSec() << " " 
                    << current_calib_(0) << " " << current_calib_(1) << " " 
                    << dynamic_sigma << " " << (dynamic_sigma * 1.5) << " " 
                    << wheel_vel_x << " " << wheel_omega << " " << (turn_rate > 0.1 ? 1 : 0) << std::endl;
            }

            if (g_logger.f_graph_error.is_open()) {
                double error = isam_->getFactorsUnsafe().error(isam_->calculateEstimate());
                g_logger.f_graph_error << std::fixed << std::setprecision(6) << now.toSec() << " " << error << " " << (error > 1e-6 ? std::sqrt(error) : 0.0) << std::endl;
            }

        } catch (...) {
            // 这样即使优化失败，机器人也能根据编码器继续更新位置，避免"卡死"在原地
            ROS_WARN("[Rescue] Resetting to ODOM PREDICTION: X=%.2f (Avoid Deadlock)", pred_pose.x());
                
            // pred_pose 是函数开头计算的：上一帧位姿 + 里程计增量
            // 这样能保证地图和激光数据依然大概率是匹配的
            hardReset(pred_pose, current_velocity_);
        }
    }
    gtsam::Pose3 getPose() { return current_pose_; }
    gtsam::Vector3 getVelocity() { return current_velocity_; }
    bool isDegenerate() const { return is_degenerate_; }
    gtsam::Matrix6 getCovariance() const { return current_covariance_; }

private:
    static Symbol X(int i) { return Symbol('x', i); }
    static Symbol V(int i) { return Symbol('v', i); }
    static Symbol C(int i) { return Symbol('c', i); } 

    gtsam::Point2 measured_uv(cv::Point2f p) { return gtsam::Point2(p.x, p.y); }

    
    double estimated_wall_y_ = 0.0; // 墙体的全局 Y 坐标估计
    bool wall_y_initialized_ = false;
    std::unique_ptr<gtsam::ISAM2> isam_;
    unsigned long key_index_;
    gtsam::Pose3 current_pose_;
    gtsam::Vector3 current_velocity_;
    ros::Time last_keyframe_time_; 
    gtsam::Pose3 last_keyframe_pose_;
    gtsam::Cal3_S2::shared_ptr K_;
    gtsam::Matrix6 current_covariance_;
    bool is_degenerate_ = false;
};

// ================= [🔴 最终修复版 V13：删除重复定义，编译通过] =================

class StructureOdometer {
    public:
        double predicted_dist_ = 0.0;      
        double wheel_speed_ = 0.0;
        double wheel_omega_ = 0.0; 
        ros::Time last_fix_time_;          
        int last_snapped_idx_ = 0;
        
        VanishingPointDetector vp_detector_;
        PeriodicityEstimator period_estimator_; 
        ExtrinsicCalibrator calibrator_; 
    
        gtsam::Vector4 latest_plane_coeffs_;
        bool has_new_plane_;
        
        bool is_lio_healthy_ = true; 
        ros::Time last_degraded_time_;
        
        double continuous_degraded_time_ = 0.0; 
        bool lio_fatal_failure_ = false;        
    
        // 全局修正角度 (8.28度)
        const double CORRIDOR_OFFSET_RAD = 0.10472; 

        gtsam::Pose3 getTrackerPose() {
            return tracker_.getPose();
        }

        void updateVisualCalibration(int u, int v, const cv::Mat& mask, cv::Mat& debug_img) {
            std::lock_guard<std::mutex> lock(odom_mutex_); 
            calibrator_.update(u, v, mask, debug_img);
        }
    
        StructureOdometer(ros::NodeHandle& nh) 
            : nh_(nh), is_active_(false), last_fix_amount_(0.0), has_new_plane_(false), 
              wall_detected_(false), front_wall_detected_(false),
              has_new_line_(false) {
            
            latest_line_res_ = {false, 0.0, 0.0, 0};
            
            pub_corrected_path_ = nh_.advertise<nav_msgs::Path>("semantic_corrected_path", 1);
            pub_corrected_odom_ = nh_.advertise<nav_msgs::Odometry>("semantic_corrected_odom", 1);
            pub_uncertainty_ = nh_.advertise<visualization_msgs::Marker>("uncertainty_ellipse", 1);
            
            sub_wheel_odom_ = nh_.subscribe("/odom", 100, &StructureOdometer::odomEncoderCallback, this);
            
            std::string lio_topic;
            nh_.param<std::string>("lio_odom_topic", lio_topic, "/Odometry"); 
            sub_lio_odom_ = nh_.subscribe(lio_topic, 100, &StructureOdometer::odomLIOCallback, this);
            
            ROS_INFO("========================================");
            ROS_INFO("[System] HYBRID FUSION MODE: ACTIVATED");
            ROS_INFO("   - Longitudinal (X): Trusted ENCODER");
            ROS_INFO("   - Lateral (Y) & Yaw: Trusted LIO (Walls constrained)");
            ROS_INFO("========================================");
            
            if (config.global_frame_id.empty()) config.global_frame_id = "camera_init";
            corrected_path_.header.frame_id = config.global_frame_id;
            
            last_proc_time_ = ros::Time::now();
            last_trigger_time_ = ros::Time(0);
            last_fix_time_ = ros::Time(0);
            last_degraded_time_ = ros::Time(0);
            
            latest_lio_pose_ = gtsam::Pose3();
            lio_to_map_offset_ = gtsam::Pose3(); 
            has_lio_data_ = false;
            was_degraded_ = false;
            lio_fatal_failure_ = false;
            continuous_degraded_time_ = 0.0;
            check_enc_integration_ = 0.0;
            
            // 初始化混合融合变量
            last_valid_lio_y_ = 0.0;
            last_valid_lio_z_ = 0.0;
    
            ROS_INFO("StructureOdometer Started.");
        }
    
        void odomEncoderCallback(const nav_msgs::Odometry::ConstPtr& msg) {
            double raw_speed = msg->twist.twist.linear.x;
            double raw_omega = msg->twist.twist.angular.z;
            if (std::abs(raw_speed) < 0.005 && std::abs(raw_omega) < 0.005) {
                enc_speed_ = 0.0; enc_omega_ = 0.0;
            } else {
                enc_speed_ = raw_speed; enc_omega_ = raw_omega;
            }
        }

        void odomLIOCallback(const nav_msgs::Odometry::ConstPtr& msg) {
            static ros::Time last_msg_time(0);
            ros::Time now = msg->header.stamp;
            double dt = (now - last_msg_time).toSec();

            gtsam::Point3 t_raw(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
            gtsam::Rot3 R_raw = gtsam::Rot3::Quaternion(
                msg->pose.pose.orientation.w, 
                msg->pose.pose.orientation.x, 
                msg->pose.pose.orientation.y, 
                msg->pose.pose.orientation.z
            );

            // 源头旋转对齐 (8.28度)
            gtsam::Rot3 R_correct = gtsam::Rot3::Yaw(CORRIDOR_OFFSET_RAD);
            gtsam::Point3 t_aligned = R_correct.rotate(t_raw);
            gtsam::Rot3 R_aligned = R_correct.compose(R_raw);

            {
                std::lock_guard<std::mutex> lock(odom_mutex_);
                
                if (!has_lio_data_) check_lio_start_pos_ = t_aligned;

                latest_lio_pose_ = gtsam::Pose3(R_aligned, t_aligned);
                
                if (dt > 0.001 && dt < 0.2) {
                    double yaw = R_aligned.yaw();
                    static gtsam::Point3 last_t_aligned = t_aligned;
                    
                    double vx_world = (t_aligned.x() - last_t_aligned.x()) / dt;
                    double vy_world = (t_aligned.y() - last_t_aligned.y()) / dt;
                    
                    lio_vel_x_body_check_ = vx_world * std::cos(yaw) + vy_world * std::sin(yaw);
                    lio_vel_y_body_check_ = -vx_world * std::sin(yaw) + vy_world * std::cos(yaw);
                    
                    last_t_aligned = t_aligned;
                }
                has_lio_data_ = true;
            }
            last_msg_time = now;
        }

        void updateGroundPlane(const gtsam::Vector4& plane) {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            latest_plane_coeffs_ = plane;
            has_new_plane_ = true;
        }
    
        void setWallDetected(bool detected) {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            wall_detected_ = detected;
        }

        void setFrontWallDetected(bool detected) {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            front_wall_detected_ = detected;
        }
    
        void updateLineFeature(const StructuralFeatureExtractor::LineResult& res) {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            has_new_line_ = res.valid;
            latest_line_res_ = res;
        }
    
        void update(const ros::Time& now, double measurement, bool vp_valid, double vp_yaw, cv::Point2f vp_pt, int vp_inliers,
            bool has_vis_vel, const gtsam::Vector3& vis_vel, double vis_conf, double avg_vis_depth) {
            std::lock_guard<std::mutex> lock(odom_mutex_);

            // ================= [1. 文件流定义] =================
            static std::ofstream f_legs("/home/jiangjiacheeng/spm_data/leg_indices.txt", std::ios::out); 
            static bool legs_header_written = false;
            static std::ofstream f_feats("/home/jiangjiacheeng/spm_data/features.txt", std::ios::out);
            static bool feats_header_written = false;

            if (f_legs.is_open() && !legs_header_written) {
                f_legs << "time leg_idx pred_dist signal_meas" << std::endl;
                legs_header_written = true;
            }
            if (f_feats.is_open() && !feats_header_written) {
                f_feats << "time measurement threshold is_triggered" << std::endl;
                feats_header_written = true;
            }

            double dt = (now - last_proc_time_).toSec();
            if (dt < 0.001) return;
            last_proc_time_ = now;

            // ================= [2. 更新速度状态] =================
            if (is_ferrying_mode_) {
                wheel_speed_ = enc_speed_; wheel_omega_ = enc_omega_;
            } else {
                wheel_speed_ = enc_speed_; wheel_omega_ = enc_omega_;
            }

            // [🔴 修复点] 必须在这里定义，确保全局可见
            bool is_moving_forward = (wheel_speed_ > 0.05);
            bool is_reversing = (wheel_speed_ < -0.05);

            // ================= [3. 激进版：原地转向锁 (Spot Turn Lock)] =================
            const double ENC_SCALE_FWD = 1.071; 
            const double ENC_SCALE_REV = 1.0477; 
            double current_scale = (wheel_speed_ >= 0.0) ? ENC_SCALE_FWD : ENC_SCALE_REV;
            
            // 判定条件：角速度大 (>0.05) 且 线速度小 (<0.15)
            bool is_turning_condition = (std::abs(wheel_omega_) > 0.05) && (std::abs(wheel_speed_) < 0.15);

            // 冷却计时器
            static double lock_timer = 0.0;
            if (is_turning_condition) {
                lock_timer = 0.3; // 重置冷却时间
            } else {
                if (lock_timer > 0.0) lock_timer -= dt; 
            }

            bool is_locked = (lock_timer > 0.0); 

            double dist_step = 0.0;
            double input_vx = 0.0; 

            if (is_locked) {
                // [锁死状态]
                dist_step = 0.0;
                input_vx = 0.0; 
            } else {
                // [正常行驶]
                dist_step = (wheel_speed_ * current_scale) * dt;
                input_vx = wheel_speed_ * current_scale;
            }
            
            // 积分预测距离
            if (std::abs(input_vx) > 0.001) { 
                predicted_dist_ += dist_step; 
            }
            // ======================================================================

            // ================= [4. LIO 健康检测] =================
            bool lio_current_stuck = false;
            
            if (std::abs(wheel_speed_) > 0.2) { 
                if (std::abs(lio_vel_x_body_check_) < 0.05) lio_current_stuck = true; 
            }
            if (std::abs(wheel_speed_) > 0.15 && std::abs(lio_vel_x_body_check_) > 0.1) {
                if (wheel_speed_ * lio_vel_x_body_check_ < 0) { 
                    lio_current_stuck = true;
                    ROS_WARN_THROTTLE(1.0, "[Degeneracy] DIRECTION MISMATCH!");
                }
            }
            if (std::abs(lio_vel_x_body_check_) > 5.0) {
                lio_current_stuck = true;
                ROS_WARN_THROTTLE(1.0, "[Degeneracy] LIO JUMP DETECTED!");
            }
            if (std::abs(wheel_speed_) > 0.05) {
                check_enc_integration_ += wheel_speed_ * dt;
                if (std::abs(check_enc_integration_) > 0.3) {
                    double lio_dist = (latest_lio_pose_.translation() - check_lio_start_pos_).norm();
                    if (lio_dist < 0.10) {
                        lio_current_stuck = true;
                        ROS_WARN_THROTTLE(1.0, "[Degeneracy] OSCILLATION DETECTED!");
                    }
                    check_enc_integration_ = 0.0;
                    check_lio_start_pos_ = latest_lio_pose_.translation();
                }
            }

            // ================= [5. 熔断逻辑] =================
            if (lio_current_stuck) continuous_degraded_time_ += dt; 
            else {
                if (continuous_degraded_time_ > 0) continuous_degraded_time_ -= dt;
                if (continuous_degraded_time_ < 0) continuous_degraded_time_ = 0;
            }

            if (continuous_degraded_time_ > 2.0) {
                if (!lio_fatal_failure_) ROS_ERROR("[System] LIO FATAL FAILURE. HYBRID RESCUE ENGAGED.");
                lio_fatal_failure_ = true;
            }

            if (lio_fatal_failure_) is_lio_healthy_ = false; 
            else {
                if (lio_current_stuck) { is_lio_healthy_ = false; last_degraded_time_ = now; } 
                else { if ((now - last_degraded_time_).toSec() > 1.0) is_lio_healthy_ = true; }
            }
            
            if (is_ferrying_mode_) is_lio_healthy_ = true; 
            if (!has_lio_data_) is_lio_healthy_ = false; 

            // ================= [6. 策略分支] =================

            if (is_lio_healthy_) {
                if (was_degraded_) {
                    lio_to_map_offset_ = latest_lio_pose_.between(tracker_.getPose());
                    was_degraded_ = false; 
                    check_enc_integration_ = 0.0;
                    check_lio_start_pos_ = latest_lio_pose_.translation();
                    ROS_WARN("[System] LIO Recovered!");
                }
                
                gtsam::Pose3 target_pose = latest_lio_pose_.compose(lio_to_map_offset_);
                tracker_.hardReset(target_pose, gtsam::Vector3(lio_vel_x_body_check_, 0, 0));
                
                is_course_locked_ = false;
                forward_yaw_samples_ = 0;
                
                last_valid_lio_y_ = target_pose.y();
                last_valid_lio_z_ = target_pose.z();
                last_valid_lio_rot_ = target_pose.rotation();

                predicted_dist_ = target_pose.x();

                // 更新 leg_idx
                double current_spacing = (period_estimator_.getConfidence() > 0.4) ? 
                                        period_estimator_.getSpacing() : config.leg_spacing;
                int current_idx = std::round(predicted_dist_ / current_spacing);
                if (current_idx != last_snapped_idx_) {
                    last_snapped_idx_ = current_idx;
                    last_trigger_time_ = now; 
                }

                if (g_logger.f_traj.is_open()) {
                    g_logger.f_traj << std::fixed << std::setprecision(6) << now.toSec() << " "
                    << target_pose.x() << " " << target_pose.y() << " " << target_pose.z() << " "
                    << target_pose.rotation().toQuaternion().x() << " " << target_pose.rotation().toQuaternion().y() << " "
                    << target_pose.rotation().toQuaternion().z() << " " << target_pose.rotation().toQuaternion().w() << " "
                    << lio_vel_x_body_check_ << " " << lio_vel_y_body_check_ << " " << 0.0 << std::endl;
                }

                publishCorrectedData(now, target_pose.x(), target_pose.y(), target_pose.z(), target_pose.rotation().toQuaternion());
                
                if (f_legs.is_open()) {
                    f_legs << std::fixed << std::setprecision(4) 
                           << now.toSec() << " " << last_snapped_idx_ << " " << predicted_dist_ << " " << measurement << std::endl;
                }
                if (f_feats.is_open()) {
                    f_feats << std::fixed << std::setprecision(4) << now.toSec() << " " << measurement << " " << 0 << " " << 0.0 << std::endl;
                }
                return; 
            } 

            // ================= [7. 救援模式逻辑] =================
            was_degraded_ = true;
            
            double input_vy = 0.0;
            double used_omega = 0.0; 

            bool add_periodicity_factor = false; 
            if (!is_ferrying_mode_) {
                // [🔴 现在 is_reversing 可以在这里被正确访问了]
                double effective_thresh = is_reversing ? (config.trigger_thresh * 0.6) : config.trigger_thresh;
                bool leg_detected = (measurement > effective_thresh);
                
                if (!is_active_ && leg_detected) { 
                    is_active_ = true; 
                } else if (is_active_ && !leg_detected) {
                    if ((now - last_trigger_time_).toSec() > config.cooldown_time) {
                        
                        double current_spacing = (period_estimator_.getConfidence() > 0.4) ? 
                                                period_estimator_.getSpacing() : config.leg_spacing;
                        
                        int closest_idx = std::round(predicted_dist_ / current_spacing);
                        double target_dist = closest_idx * current_spacing;
                        double dist_error = std::abs(predicted_dist_ - target_dist);
                        
                        if (dist_error < (current_spacing * 0.45)) { 
                            add_periodicity_factor = true;
                            last_snapped_idx_ = closest_idx;
                            last_trigger_time_ = now;
                            predicted_dist_ = 0.2 * predicted_dist_ + 0.8 * target_dist; 
                            ROS_INFO("[Grid] SNAP! Idx: %d, Dist: %.2f -> %.2f", closest_idx, predicted_dist_, target_dist);
                        } else {
                            ROS_WARN("[Grid] SNAP REJECTED! Error too large: %.3fm", dist_error);
                        }
                    }
                    is_active_ = false;
                }
            }
            
            StructuralFeatureExtractor::LineResult corrected_line_res = latest_line_res_; 
            if (vp_valid && has_new_line_ && std::abs(input_vx) > 0.05) {
                calibrator_.updateJointly(vp_yaw, latest_line_res_.angle);
            }
            gtsam::Rot3 calib_correction = calibrator_.getCorrectionRotation();
            if (has_new_line_) corrected_line_res.angle -= calib_correction.yaw();

            tracker_.predictAndUpdate(now, add_periodicity_factor, 1.66, 0.0, 
                vp_valid, vp_yaw, vp_pt, 0.0, 
                calib_correction, 
                input_vx, used_omega, 
                has_new_plane_, latest_plane_coeffs_, 
                wall_detected_, front_wall_detected_, 
                has_vis_vel, vis_vel, vis_conf, avg_vis_depth, 
                has_new_line_, corrected_line_res,
                false, 0.0, gtsam::Point3(), 
                is_ferrying_mode_,
                input_vy); 

            if (g_logger.f_calib.is_open()) {
                auto bias = calibrator_.getCurrentBias(); 
                g_logger.f_calib << std::fixed << std::setprecision(5) << now.toSec() << " " << bias.first << " " << bias.second << std::endl;
            }
            
            if(has_new_plane_) has_new_plane_ = false; 
            has_new_line_ = false; 
            wall_detected_ = false; 
            front_wall_detected_ = false;

            gtsam::Pose3 tracker_pose = tracker_.getPose();
            float final_x = tracker_pose.x();
            float final_y = latest_lio_pose_.y(); 
            float final_z = latest_lio_pose_.z();
            gtsam::Quaternion final_q = latest_lio_pose_.rotation().toQuaternion();

            if (std::abs(final_y - last_valid_lio_y_) > 0.5) {
                ROS_WARN_THROTTLE(1.0, "[Hybrid] LIO Lateral Jump detected! Fallback to Lock.");
                final_y = last_valid_lio_y_; 
            } else {
                last_valid_lio_y_ = final_y;
                last_valid_lio_z_ = final_z;
                last_valid_lio_rot_ = latest_lio_pose_.rotation();
            }

            if (f_legs.is_open()) {
                f_legs << std::fixed << std::setprecision(4) 
                       << now.toSec() << " " << last_snapped_idx_ << " " << predicted_dist_ << " " << measurement << std::endl;
            }

            if (f_feats.is_open()) {
                f_feats << std::fixed << std::setprecision(4) 
                        << now.toSec() << " " << measurement << " " << config.trigger_thresh << " " << (add_periodicity_factor ? 1.0 : 0.0) << std::endl;
            }

            publishCorrectedData(now, final_x, final_y, final_z, final_q);
        }
            
        void drawVisualization(cv::Mat& img, double meas, const VanishingPointDetector::VPResult& vp_res) {
            int h = img.rows; int w = img.cols;
            int roi_h = h * config.roi_h_ratio; int roi_w = w * config.roi_w_ratio; int roi_y = h - roi_h - 5;   
            cv::Rect roi_L(5, roi_y, roi_w, roi_h); cv::Rect roi_R(w - roi_w - 5, roi_y, roi_w, roi_h);
            
            cv::Scalar col = (meas > config.trigger_thresh) ? cv::Scalar(0,0,255) : cv::Scalar(255,0,0);
            int th = (meas > config.trigger_thresh) ? 3 : 2;
            cv::rectangle(img, roi_L, col, th); cv::rectangle(img, roi_R, col, th);
    
            if (vp_res.valid) {
                // 画出指向 VP 的紫色引导线
                cv::line(img, cv::Point(img.cols/2, img.rows), vp_res.pt, cv::Scalar(255, 0, 255), 2); 
                
                // [🔴 修复] 将这两行加回来，画出 VP 点
                cv::circle(img, vp_res.pt, 6, cv::Scalar(0, 255, 255), -1); // 黄色实心点 (VP位置)
                cv::circle(img, vp_res.pt, 10, cv::Scalar(0, 0, 255), 2);   // 红色空心圆环 (醒目提示)
            }
    
            int line_y = 30;
            auto drawText = [&](std::string label, std::string val, cv::Scalar color) {
                std::string full_txt = label + ": " + val;
                cv::putText(img, full_txt, cv::Point(12, line_y), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(0,0,0), 4);
                cv::putText(img, full_txt, cv::Point(12, line_y), cv::FONT_HERSHEY_PLAIN, 1.3, color, 2);
                line_y += 24;
            };
    
            std::string status_txt;
            cv::Scalar status_col;
            if (lio_fatal_failure_) { status_txt = "LIO: FATAL (HYBRID)"; status_col = cv::Scalar(0, 0, 255); }
            else if (is_lio_healthy_) { status_txt = "LIO: HEALTHY"; status_col = cv::Scalar(0, 255, 0); }
            else { status_txt = "LIO: DEGRADED"; status_col = cv::Scalar(0, 255, 255); }
            cv::putText(img, status_txt, cv::Point(w/2 - 80, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2);

            drawText("=== VISION (VP)", "", cv::Scalar(255, 255, 255));
            if (vp_res.valid) {
                double yaw_deg = vp_res.yaw * 180.0 / CV_PI;
                drawText("Status", "LOCKED (" + std::to_string(vp_res.inliers) + ")", cv::Scalar(0, 255, 0));
                drawText("Yaw", std::to_string(yaw_deg).substr(0, 5) + " deg", cv::Scalar(0, 255, 255));
            } else {
                drawText("Status", "SEARCHING...", cv::Scalar(100, 100, 100));
            }
    
            line_y += 8;
            drawText("=== POSITION", "", cv::Scalar(255, 255, 255));
            drawText("Map X", std::to_string(tracker_.getPose().x()).substr(0, 5) + " m", cv::Scalar(0, 255, 0));
            std::string leg_str = "#" + std::to_string(last_snapped_idx_);
            cv::Scalar leg_color = (meas > config.trigger_thresh) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 165, 255);
            drawText("Leg Idx", leg_str, leg_color); 
    
            double adaptive_spacing = period_estimator_.getSpacing();
            double conf = period_estimator_.getConfidence();
            cv::Scalar conf_color = (conf > 0.4) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            std::string spacing_str = "Spacing: " + std::to_string(adaptive_spacing).substr(0, 4) + "m";
            std::string conf_str = "Conf: " + std::to_string(int(conf * 100)) + "%";
    
            drawText("=== ADAPTIVE", "", cv::Scalar(255, 255, 255));
            drawText(spacing_str, conf_str, conf_color);
            drawText("=== CALIB", "", cv::Scalar(255, 255, 255));
            drawText("State", calibrator_.getStatusString(), cv::Scalar(0, 255, 255));
            cv::putText(img, "Vis: Full Visual", cv::Point(img.cols-220, img.rows-20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
        }
    
        void publishCorrectedData(const ros::Time& stamp, float x, float y, float z, gtsam::Quaternion q) {
            static std::ofstream save_file("/home/jiangjiacheeng/spm_data/final_rotated_traj.txt", std::ios::out);
            if (save_file.is_open()) {
                save_file << std::fixed << std::setprecision(6) << stamp.toSec() << " "
                        << x << " " << y << " " << z << " "
                        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            }

            nav_msgs::Odometry odom;
            odom.header.stamp = stamp; 
            odom.header.frame_id = config.global_frame_id;
            odom.child_frame_id = "body_corrected";
            odom.pose.pose.position.x = x; 
            odom.pose.pose.position.y = y; 
            odom.pose.pose.position.z = z;
            odom.pose.pose.orientation.x = q.x(); 
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z(); 
            odom.pose.pose.orientation.w = q.w();
            pub_corrected_odom_.publish(odom);
            
            geometry_msgs::PoseStamped pose;
            pose.header = odom.header; 
            pose.pose = odom.pose.pose;
            corrected_path_.header.stamp = stamp;
            corrected_path_.poses.push_back(pose);
            pub_corrected_path_.publish(corrected_path_);

            if (pub_uncertainty_.getNumSubscribers() > 0) {
                bool show_as_healthy = is_lio_healthy_ && !lio_fatal_failure_;
                if (show_as_healthy) {
                     visualization_msgs::Marker ellipse;
                     ellipse.header.frame_id = config.global_frame_id;
                     ellipse.header.stamp = stamp;
                     ellipse.ns = "uncertainty"; ellipse.id = 0;
                     ellipse.type = visualization_msgs::Marker::SPHERE;
                     ellipse.action = visualization_msgs::Marker::ADD;
                     ellipse.pose.position.x = x; 
                     ellipse.pose.position.y = y; 
                     ellipse.pose.position.z = z;
                     ellipse.scale.x = 0.1; ellipse.scale.y = 0.1; ellipse.scale.z = 0.1;
                     ellipse.color.r = 0.0; ellipse.color.g = 1.0; ellipse.color.b = 0.0; ellipse.color.a = 0.8;
                     pub_uncertainty_.publish(ellipse);
                } else {
                    gtsam::Matrix6 cov = tracker_.getCovariance();
                    gtsam::Matrix3 pos_cov = cov.block<3, 3>(3, 3); 
                    Eigen::SelfAdjointEigenSolver<gtsam::Matrix3> eigensolver(pos_cov);
                    if (eigensolver.info() == Eigen::Success) {
                        visualization_msgs::Marker ellipse;
                        ellipse.header.frame_id = config.global_frame_id;
                        ellipse.header.stamp = stamp;
                        ellipse.ns = "uncertainty"; ellipse.id = 0;
                        ellipse.type = visualization_msgs::Marker::SPHERE;
                        ellipse.action = visualization_msgs::Marker::ADD;
                        ellipse.pose.position.x = x;
                        ellipse.pose.position.y = y;
                        ellipse.pose.position.z = z;
                        Eigen::Quaterniond q_eig(eigensolver.eigenvectors());
                        ellipse.pose.orientation.x = q_eig.x(); ellipse.pose.orientation.y = q_eig.y();
                        ellipse.pose.orientation.z = q_eig.z(); ellipse.pose.orientation.w = q_eig.w();
                        Eigen::Vector3d val = eigensolver.eigenvalues();
                        ellipse.scale.x = 3.0 * std::sqrt(std::max(0.001, val(0))); 
                        ellipse.scale.y = 3.0 * std::sqrt(std::max(0.001, val(1)));
                        ellipse.scale.z = 3.0 * std::sqrt(std::max(0.001, val(2)));
                        if (lio_fatal_failure_) {
                            ellipse.color.r = 1.0; ellipse.color.g = 0.0; ellipse.color.b = 0.0; ellipse.color.a = 0.8;
                        } else {
                            ellipse.color.r = 1.0; ellipse.color.g = 1.0; ellipse.color.b = 0.0; ellipse.color.a = 0.5;
                        }
                        pub_uncertainty_.publish(ellipse);
                    }
                }
            }
        }
    
        ros::NodeHandle nh_;
        ros::Publisher pub_corrected_path_, pub_corrected_odom_;
        ros::Subscriber sub_wheel_odom_; 
        nav_msgs::Path corrected_path_;
        FactorGraphTracker tracker_;
        ros::Time last_proc_time_;
        bool is_active_;
        ros::Time last_trigger_time_;
        double last_fix_amount_;
        
    private:
        std::mutex odom_mutex_; 
        bool wall_detected_;       
        bool front_wall_detected_; 
        bool is_ferrying_mode_ = false;

        double enc_speed_ = 0.0;
        double enc_omega_ = 0.0;

        gtsam::Pose3 latest_lio_pose_;
        gtsam::Pose3 lio_to_map_offset_; 
        bool has_lio_data_ = false;
        double lio_vel_x_body_check_ = 0.0; 
        double lio_vel_y_body_check_ = 0.0; 
        double lio_inst_speed_ = 0.0;
        double last_y_check_ = 0.0;
        bool was_degraded_ = false; 

        double accum_yaw_sum_ = 0.0;   
        double accum_dist_sum_ = 0.0;  
        int accum_seq_ = 0;            
        ros::Subscriber sub_lio_odom_;   

        StructuralFeatureExtractor::LineResult latest_line_res_;
        bool has_new_line_;
        ros::Publisher pub_uncertainty_;

        double avg_forward_yaw_ = 0.0;     
        double avg_forward_y_ = 0.0;
        double rail_y_offset_ = 0.0;
        int forward_yaw_samples_ = 0;      
        bool is_course_locked_ = false;    
        double locked_yaw_ = 0.0;          

        gtsam::Point3 locked_anchor_ = gtsam::Point3(0,0,0);
        
        gtsam::Point3 check_lio_start_pos_;
        double check_enc_integration_ = 0.0;

        // 混合融合缓存
        double last_valid_lio_y_ = 0.0;
        double last_valid_lio_z_ = 0.0;
        gtsam::Rot3 last_valid_lio_rot_;
};
    class SemanticOctomapServer {
        public:
            SemanticOctomapServer() : smoothed_a_(0.0), smoothed_b_(0.0), smoothed_c_(0.0), semantic_odom_(nh_) {
                ros::NodeHandle nh_private("~"); 
                loadParameters(nh_private);
                nh_ = ros::NodeHandle();
                
                octree_ = new octomap::ColorOcTree(0.05);
                octree_->setProbHit(0.7); octree_->setProbMiss(0.4); 
                octree_->setClampingThresMin(config.log_odds_min); octree_->setClampingThresMax(config.log_odds_max);
        
                latest_high_intensity_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
               
                verify_radius_ = 0.3;          
                metal_intensity_thresh_ = 0.5; 
        
                // [NEW] 初始化流程图可视化话题
                pub_viz_step1_height_ = nh_.advertise<sensor_msgs::Image>("process/1_geometric_height", 1);
                pub_viz_step2_hsv_ = nh_.advertise<sensor_msgs::Image>("process/2_color_rejection", 1);
                pub_viz_step3_morph_ = nh_.advertise<sensor_msgs::Image>("process/3_morphological", 1);
                pub_viz_step4_lidar_ = nh_.advertise<sensor_msgs::Image>("process/4_cross_modal_fusion", 1);
                pub_2d_map_ = nh_.advertise<nav_msgs::OccupancyGrid>("projected_map", 1, true);
                pub_debug_img_ = nh_.advertise<sensor_msgs::Image>("traversability_debug", 1);
                pub_vp_debug_ = nh_.advertise<sensor_msgs::Image>("vp_debug_rgb", 1);
                pub_marker_ = nh_.advertise<visualization_msgs::Marker>("nav_center_line", 1);

                pub_ground_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("ground_cloud_viz", 1);
      
                pub_vis_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("semantic_octomap_markers", 1);
                sub_lidar_ = nh_.subscribe("/cloud_registered_effect_world", 1, &SemanticOctomapServer::lidarCallback, this);
        
                sub_depth_filter_ = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "/front_camera/aligned_depth_to_color/image_raw", 1);
                sub_rgb_filter_   = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "/front_camera/color/image_raw", 1);
                sync_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *sub_depth_filter_, *sub_rgb_filter_);
                sync_->registerCallback(boost::bind(&SemanticOctomapServer::syncCallback, this, _1, _2));
        
                last_vis_time_ = ros::Time(0); 
                voxel_filter_.setLeafSize(0.05f, 0.05f, 0.05f); // 5cm 分辨率
                ROS_INFO("Semantic Navigator Started!");
            }
        
            ~SemanticOctomapServer() { 
                std::cout << "\n\n[System] Node shutting down. Attempting to save map..." << std::endl;
                
                // 强制保存
                saveMap(); 

                sub_lidar_.shutdown(); 
                delete sync_; 
                delete sub_depth_filter_; 
                delete sub_rgb_filter_; 
                delete octree_;
                
                std::cout << "[System] Shutdown complete.\n" << std::endl;
            }
        
            typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
        
            float colorDist(const cv::Vec3b& a, const cv::Vec3b& b) {
                float dH = std::abs(a[0] - b[0]);
                if (dH > 90) dH = 180 - dH; 
                return std::sqrt(dH*dH*2.0 + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2])*0.5);
            }
        
            double calculateOtsuThreshold(const std::vector<float>& intensities) {
                if (intensities.empty()) return 0.5;
                float max_val = 0;
                for(float v : intensities) if(v > max_val) max_val = v;
                if(max_val < 0.1) return 0.5; 
        
                int hist_size = 256;
                std::vector<int> hist(hist_size, 0);
                for(float v : intensities) {
                    int bin = std::min((int)(v / max_val * 255), 255);
                    hist[bin]++;
                }
        
                float total = intensities.size();
                float sum = 0;
                for(int t=0; t<hist_size; t++) sum += t * hist[t];
        
                float sumB = 0;
                int wB = 0;
                int wF = 0;
                float varMax = 0;
                int threshold = 0;
        
                for(int t=0; t<hist_size; t++) {
                    wB += hist[t];
                    if(wB == 0) continue;
                    wF = total - wB;
                    if(wF == 0) break;
        
                    sumB += (float)(t * hist[t]);
                    float mB = sumB / wB;
                    float mF = (sum - sumB) / wF;
        
                    float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
                    if(varBetween > varMax) {
                        varMax = varBetween;
                        threshold = t;
                    }
                }
                return (float)threshold / 255.0 * max_val;
            }
        
            bool isDynamicObject(const octomap::point3d& sensor_origin, const octomap::point3d& measurement) {
                octomap::point3d direction = measurement - sensor_origin;
                octomap::point3d hit_point;
                if (octree_->castRay(sensor_origin, direction, hit_point, true, 10.0)) {
                    double dist_map = sensor_origin.distance(hit_point);
                    double dist_meas = sensor_origin.distance(measurement);
                    if (dist_meas < dist_map - config.dynamic_dist_diff) {
                        return true;
                    }
                }
                return false;
            }
        
            // [🔴 进阶玩法] 阶段 1：图像处理与掩膜生成
            void syncCallback(const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg) {
                // 1. 转换图像
                cv_bridge::CvImagePtr cv_rgb_ptr;
                try { 
                    cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8); 
                } catch (...) { return; }
                
                cv::Mat rgb_img = cv_rgb_ptr->image; 
                
                // 2. 获取当前帧相机的绝对位姿 (直接信任 Faster-LIO 的 TF)
                tf::StampedTransform transform;
                try {
                    if (!tf_listener_.canTransform(config.global_frame_id, depth_msg->header.frame_id, depth_msg->header.stamp)) return;
                    tf_listener_.lookupTransform(config.global_frame_id, depth_msg->header.frame_id, depth_msg->header.stamp, transform);
                } catch (...) { return; }

                // 转换 TF 到 GTSAM Pose3 (World -> Camera)
                gtsam::Pose3 T_w_c(
                    gtsam::Rot3(gtsam::Quaternion(transform.getRotation().w(), transform.getRotation().x(), transform.getRotation().y(), transform.getRotation().z())),
                    gtsam::Point3(transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z())
                );
                
                // 3. 图像处理：提取红色的笼子腿 (复用你之前的逻辑)
                // 简单起见，我们直接用颜色提取，不再做繁琐的深度校验，因为点云投影会负责深度校验
                cv::Mat hsv_img, mask_red1, mask_red2, cage_mask;
                cv::cvtColor(rgb_img, hsv_img, cv::COLOR_BGR2HSV);
                
                // 提取红色 (HSV)
                cv::inRange(hsv_img, cv::Scalar(0, config.hsv_s_min, config.hsv_v_min), cv::Scalar(10, 255, 255), mask_red1);
                cv::inRange(hsv_img, cv::Scalar(170, config.hsv_s_min, config.hsv_v_min), cv::Scalar(180, 255, 255), mask_red2);
                cv::bitwise_or(mask_red1, mask_red2, cage_mask);

                // 形态学操作：去噪
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                // [🔴 修改] 针对镂空笼子的“补洞”策略
                // 1. 先做开运算去噪 (去掉孤立噪点)
                cv::Mat kernel_noise = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                cv::morphologyEx(cage_mask, cage_mask, cv::MORPH_OPEN, kernel_noise);

                // 2. [核心] 使用巨大的核进行“闭运算 (CLOSE)”
                // 闭运算 = 先膨胀后腐蚀。它能填平物体内部的小黑洞，但保持物体轮廓不变。
                // Size(9, 9) 意味着能填补 9 像素宽的网格空隙。如果网眼更大，可以设为 (15, 15)
                cv::Mat kernel_fill = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)); 
                cv::morphologyEx(cage_mask, cage_mask, cv::MORPH_CLOSE, kernel_fill);
                
                // 3. 再稍微膨胀一点点，增加容错
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); 
                cv::morphologyEx(cage_mask, cage_mask, cv::MORPH_DILATE, kernel_dilate);

                // 4. [核心] 将掩膜和位姿存入全局变量，供 Lidar 线程使用
                {
                    std::lock_guard<std::mutex> lock(viz_mutex_);
                    latest_cage_mask_ = cage_mask.clone();
                    latest_cam_pose_ = T_w_c;
                    has_new_image_ = true;
                }

                // 5. 可视化一下，让自己放心
                if (pub_debug_img_.getNumSubscribers() > 0) {
                     cv::Mat debug_viz;
                     rgb_img.copyTo(debug_viz, cage_mask); // 只显示提取出来的部分
                     pub_debug_img_.publish(cv_bridge::CvImage(depth_msg->header, "bgr8", debug_viz).toImageMsg());
                }
            }
        
            pcl::PointCloud<pcl::PointXYZI>::Ptr filterHighIntensityAdaptive(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
                pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                if (cloud->empty()) return out_cloud;
        
                const int SECTORS = 12;
                std::vector<std::vector<float>> sector_intensities(SECTORS);
                std::vector<std::vector<int>> sector_indices(SECTORS);
        
                for (size_t i = 0; i < cloud->points.size(); ++i) {
                    const auto& p = cloud->points[i];
                    float angle = std::atan2(p.y, p.x); 
                    int idx = std::floor((angle + M_PI) / (2.0 * M_PI) * SECTORS);
                    idx = std::max(0, std::min(SECTORS - 1, idx));
                    sector_intensities[idx].push_back(p.intensity);
                    sector_indices[idx].push_back(i);
                }
        
                for (int i = 0; i < SECTORS; ++i) {
                    if (sector_intensities[i].empty()) continue;
                    float local_thresh = calculateOtsuThreshold(sector_intensities[i]);
                    local_thresh = std::max(5.0f, std::min(100.0f, local_thresh));
                    float final_thresh = local_thresh * 0.65;
                    for (int point_idx : sector_indices[i]) {
                        if (cloud->points[point_idx].intensity > final_thresh) {
                            out_cloud->points.push_back(cloud->points[point_idx]);
                        }
                    }
                }
                return out_cloud;
            }
        
            pcl::PointCloud<pcl::PointXYZI>::Ptr filterNoiseByClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
                pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                if (cloud->empty()) return out_cloud;
        
                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_for_clustering(new pcl::PointCloud<pcl::PointXYZI>());
                
                for (const auto& p : cloud->points) {
                    if (p.intensity > 80.0) {
                        out_cloud->points.push_back(p);
                    } else {
                        cloud_for_clustering->points.push_back(p);
                    }
                }
        
                if (cloud_for_clustering->empty()) return out_cloud;
        
                pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
                tree->setInputCloud(cloud_for_clustering);
        
                std::vector<pcl::PointIndices> cluster_indices;
                pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
                
                ec.setClusterTolerance(0.60); 
                ec.setMinClusterSize(2);      
                ec.setMaxClusterSize(500);
                ec.setSearchMethod(tree);
                ec.setInputCloud(cloud_for_clustering);
                ec.extract(cluster_indices);
        
                for (const auto& indices : cluster_indices) {
                    float min_z = 100.0, max_z = -100.0;
                    float min_x = 100.0, max_x = -100.0;
                    float min_y = 100.0, max_y = -100.0;
        
                    for (int idx : indices.indices) {
                        const auto& p = cloud_for_clustering->points[idx];
                        if (p.z < min_z) min_z = p.z;
                        if (p.z > max_z) max_z = p.z;
                        if (p.x < min_x) min_x = p.x;
                        if (p.x > max_x) max_x = p.x;
                        if (p.y < min_y) min_y = p.y;
                        if (p.y > max_y) max_y = p.y;
                    }
        
                    float height = max_z - min_z;
                    float width_x = max_x - min_x;
                    float width_y = max_y - min_y;
                    float xy_span = std::sqrt(width_x*width_x + width_y*width_y);
        
                    bool is_ground_noise = (height < 0.05 && xy_span > 0.3);
                    
                    if (!is_ground_noise) {
                         for (int idx : indices.indices) {
                             out_cloud->points.push_back(cloud_for_clustering->points[idx]);
                         }
                    }
                }
                
                return out_cloud;
            }
        
            // [🏆 最终修复版 V4] lidarCallback
            // 特性：强制过道净空 (Corridor Keep-out Zone)
            // 逻辑：在定义的过道宽度内，只有红色点能存活，灰色点全部抹杀。
            void lidarCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
                
                pcl::PointCloud<pcl::PointXYZI> input_cloud;
                pcl::fromROSMsg(*msg, input_cloud);
                if (input_cloud.empty()) return;

                cv::Mat mask;
                gtsam::Pose3 T_w_c; 
                bool data_ready = false;
                {
                    std::lock_guard<std::mutex> lock(viz_mutex_);
                    if (has_new_image_) {
                        mask = latest_cage_mask_.clone(); 
                        T_w_c = latest_cam_pose_;
                        data_ready = true;
                    }
                }
                if (!data_ready) return;

                float fx = config.cam_fx, fy = config.cam_fy;
                float cx = config.cam_cx, cy = config.cam_cy;
                gtsam::Pose3 T_c_w = T_w_c.inverse(); 

                std::vector<float> hit_depths;
                hit_depths.reserve(2000);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_scan(new pcl::PointCloud<pcl::PointXYZRGB>());
                
                bool assume_input_is_body_frame = true; 
                double current_v = semantic_odom_.wheel_speed_; 

                // ================= [🛡️ 过道净空参数配置] =================
                
                // 1. [过道物理半宽]：你的过道约 1.3m -> 半宽 0.65m
                // 我们留一点余量，设为 0.55m。
                // 含义：机器人中心左右 0.55m (总宽1.1m) 内，绝不允许出现灰色杂点。
                double corridor_clean_half_width = 0.55; 

                // 2. [红色保留阈值]：红色点只要不打到车身上(0.25m)就保留
                // 这样即使笼子有点歪，伸进过道里了，也能画出来。
                double red_min_dist = 0.25; 

                // 3. [灰色限高]：过滤头顶灯泡 (1.6m)
                double gray_height_limit = 1.60;
                
                // 4. 起点黑洞半径
                double start_exclusion_radius = 1.5; 
                // =========================================================

                for (const auto& p : input_cloud.points) {
                    if (p.z > config.z_ceiling) continue;

                    // 1. 计算世界坐标
                    gtsam::Point3 pt_w(p.x, p.y, p.z);
                    
                    // [🥊 原点黑洞] 过滤起点附近的喂料机
                    float dist_from_origin = std::sqrt(pt_w.x()*pt_w.x() + pt_w.y()*pt_w.y());
                    if (dist_from_origin < start_exclusion_radius) continue; 

                    // [🎨 地面绿化]
                    if (p.z < 0.15 && p.z > -1.0) {
                        pcl::PointXYZRGB p_ground;
                        p_ground.x = p.x; p_ground.y = p.y; p_ground.z = p.z;
                        p_ground.r = 30; p_ground.g = 180; p_ground.b = 30; 
                        colored_scan->points.push_back(p_ground);
                        continue; 
                    }

                    // 2. 转换到相机/车体坐标系 (用于计算横向偏差)
                    gtsam::Point3 pt_c = T_c_w.transformTo(pt_w); 
                    
                    // abs_y 就是点距离过道中轴线（车体中心）的横向距离
                    float abs_y = std::abs(pt_c.y()); 

                    // 3. 投影到图像
                    if (p.z < config.lidar_z_min) continue; 

                    pcl::PointXYZRGB p_obj;
                    p_obj.x = p.x; p_obj.y = p.y; p_obj.z = p.z;

                    float z_opt, x_opt, y_opt;
                    if (assume_input_is_body_frame) {
                        z_opt = pt_c.x(); x_opt = -pt_c.y(); y_opt = -pt_c.z(); 
                    } else {
                        z_opt = pt_c.z(); x_opt = pt_c.x(); y_opt = pt_c.y();
                    }

                    if (z_opt < 0.2) continue; 

                    int u = std::round(x_opt * fx / z_opt + cx);
                    int v = std::round(y_opt * fy / z_opt + cy);

                    bool is_cage = false;
                    if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows) {
                        int r = 1; 
                        for(int dy = -r; dy <= r; ++dy) {
                            for(int dx = -r; dx <= r; ++dx) {
                                int uu = std::max(0, std::min(mask.cols-1, u+dx));
                                int vv = std::max(0, std::min(mask.rows-1, v+dy));
                                if (mask.at<uchar>(vv, uu) > 128) {
                                    is_cage = true; goto hit_done;
                                }
                            }
                        }
                        hit_done:;
                    }

                    // ================= [🧠 核心逻辑：过道净空协议] =================
                    
                    if (is_cage) {
                        // 【红色特权】：允许进入过道区域
                        // 只要不打到车 ( > 0.25m )，即使它在 clean_zone 里面也保留。
                        // 这解决了“笼子歪了画不出来”的问题。
                        if (abs_y < red_min_dist) continue; 

                        p_obj.r = 255; p_obj.g = 0; p_obj.b = 0; 
                        if (z_opt > 0.5 && z_opt < 5.0) {
                            hit_depths.push_back(z_opt);
                        }
                        colored_scan->points.push_back(p_obj);
                    } 
                    else {
                        // 【灰色严管】：必须滚出过道！
                        // 既然你不是笼子，如果你出现在了 corridor_clean_half_width (0.55m) 范围内，
                        // 我不管你是操作员、苍蝇还是误差，直接删除。
                        if (abs_y < corridor_clean_half_width) continue; 
                        
                        // 高度限制 (1.6m)
                        if (p.z > gray_height_limit) continue;

                        p_obj.r = 100; p_obj.g = 100; p_obj.b = 100; 
                        colored_scan->points.push_back(p_obj);
                    }
                }

                // ==========================================
                // 阶段 2: 智能补全 (同步应用净空协议)
                // ==========================================
                if (hit_depths.size() > 10) { 
                    double sum = std::accumulate(hit_depths.begin(), hit_depths.end(), 0.0);
                    double mean = sum / hit_depths.size();
                    double sq_sum = std::inner_product(hit_depths.begin(), hit_depths.end(), hit_depths.begin(), 0.0);
                    float depth_std_dev = std::sqrt(sq_sum / hit_depths.size() - mean * mean);

                    if (depth_std_dev < 0.35) {
                        std::nth_element(hit_depths.begin(), hit_depths.begin() + hit_depths.size()/2, hit_depths.end());
                        float cage_plane_depth = hit_depths[hit_depths.size()/2];

                        if (cage_plane_depth > 0.5 && cage_plane_depth < 4.5) {
                            int step = 3; 
                            for (int v = 0; v < mask.rows; v += step) {
                                for (int u = 0; u < mask.cols; u += step) {
                                    if (mask.at<uchar>(v, u) > 128) {
                                        float x_opt = (u - cx) * cage_plane_depth / fx;
                                        float y_opt = (v - cy) * cage_plane_depth / fy;
                                        float z_opt = cage_plane_depth;

                                        gtsam::Point3 pt_c;
                                        if (assume_input_is_body_frame) pt_c = gtsam::Point3(z_opt, -x_opt, -y_opt);
                                        else pt_c = gtsam::Point3(x_opt, y_opt, z_opt);
                                        
                                        // 补全点默认是红色的，所以使用宽松的 red_min_dist
                                        if (std::abs(pt_c.y()) < red_min_dist) continue;

                                        gtsam::Point3 pt_w = T_w_c.transformFrom(pt_c);
                                        
                                        float dist_from_origin = std::sqrt(pt_w.x()*pt_w.x() + pt_w.y()*pt_w.y());
                                        if (dist_from_origin < start_exclusion_radius) continue;

                                        if (pt_w.z() < 0.15) continue; 

                                        pcl::PointXYZRGB ghost_p;
                                        ghost_p.x = pt_w.x(); ghost_p.y = pt_w.y(); ghost_p.z = pt_w.z();
                                        ghost_p.r = 255; ghost_p.g = 0; ghost_p.b = 0; 
                                        colored_scan->points.push_back(ghost_p);
                                    }
                                }
                            }
                        }
                    }
                }

                // ==========================================
                // 阶段 3: 合并 (保持不变)
                // ==========================================
                {
                    std::unique_lock<std::shared_mutex> map_lock(map_mutex_); 
                    for (const auto& p : colored_scan->points) {
                        VoxelKey key;
                        key.x = std::floor(p.x / map_resolution_);
                        key.y = std::floor(p.y / map_resolution_);
                        key.z = std::floor(p.z / map_resolution_);
                        
                        bool is_semantic = (p.r > 200) || (p.g > 150) || (p.b > 200); 
                        auto it = global_voxel_map_.find(key);
                        if (it == global_voxel_map_.end()) {
                            global_voxel_map_[key] = p;
                        } else {
                            if (is_semantic) it->second = p; 
                            else {
                                bool old_is_semantic = (it->second.r > 200) || (it->second.g > 150) || (it->second.b > 200);
                                if (!old_is_semantic) it->second = p; 
                            }
                        }
                    }

                    static int pub_counter = 0;
                    static int save_counter = 0;

                    if (pub_counter++ % 2 == 0) { 
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
                        output_cloud->reserve(global_voxel_map_.size());
                        for (const auto& kv : global_voxel_map_) output_cloud->points.push_back(kv.second);
                        
                        if (!output_cloud->empty()) {
                            sensor_msgs::PointCloud2 output_msg;
                            pcl::toROSMsg(*output_cloud, output_msg);
                            output_msg.header = msg->header;
                            output_msg.header.frame_id = config.global_frame_id;
                            pub_ground_cloud_.publish(output_msg); 
                        }
                    }

                    if (save_counter++ > 200) {
                        save_counter = 0;
                        map_lock.unlock(); 
                        saveMap();         
                        map_lock.lock();   
                    }
                }
            }


            // [💾 调试增强版] saveMap - 支持 .ot 和 .pcd 双格式保存
            void saveMap() {
                // 加锁
                std::shared_lock<std::shared_mutex> map_lock(map_mutex_); 

                // 1. 打印当前地图点数，帮你确认是否有数据
                std::cout << "\n[System] Checking map status... Points: " << global_voxel_map_.size() << std::endl;

                if (global_voxel_map_.empty()) {
                    // ⚠️ 重点：如果看到这句话，说明机器人没走出“黑洞”，或者没扫到有效点
                    std::cout << "\033[1;33m[Warning] Map is EMPTY! (Did robot move past the 2.5m exclusion zone?)\033[0m" << std::endl;
                    return;
                }

                std::cout << "[System] Saving map to disk..." << std::endl;

                // ================= [保存 OctoMap (.ot)] =================
                octomap::ColorOcTree tree(0.05); // 0.05m 分辨率

                // 准备 PCD 点云容器 (用于保存 PCD)
                pcl::PointCloud<pcl::PointXYZRGB> cloud_to_save;
                cloud_to_save.width = global_voxel_map_.size();
                cloud_to_save.height = 1;
                cloud_to_save.is_dense = false;
                cloud_to_save.points.reserve(global_voxel_map_.size());

                for (const auto& kv : global_voxel_map_) {
                    const auto& p = kv.second;
                    
                    // 填充 OctoMap
                    tree.updateNode(p.x, p.y, p.z, true); 
                    tree.setNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);

                    // 填充 PCD
                    cloud_to_save.points.push_back(p);
                }

                tree.updateInnerOccupancy();
                tree.prune(); 

                // 保存 .ot 文件
                std::string filename_ot = "/home/jiangjiacheeng/spm_data/semantic_map.ot";
                std::ofstream outfile(filename_ot, std::ios_base::out | std::ios_base::binary);
                if (outfile.is_open()) {
                    tree.write(outfile);
                    outfile.close();
                    std::cout << "\033[1;32m[Success] OctoMap SAVED to " << filename_ot << " (Nodes: " << tree.size() << ")\033[0m" << std::endl;
                } else {
                    std::cout << "\033[1;31m[Error] Failed to save .ot file " << filename_ot << "\033[0m" << std::endl;
                }

                // ================= [保存 PCD (.pcd)] =================
                std::string filename_pcd = "/home/jiangjiacheeng/spm_data/semantic_map.pcd";
                try {
                    // 使用二进制模式保存，体积更小，读写更快
                    pcl::io::savePCDFileBinary(filename_pcd, cloud_to_save);
                    std::cout << "\033[1;32m[Success] PointCloud SAVED to " << filename_pcd << " (Points: " << cloud_to_save.size() << ")\033[0m" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "\033[1;31m[Error] Failed to save .pcd file: " << e.what() << "\033[0m" << std::endl;
                }
            }
                
        
            cv::Mat applyLowLightEnhancement(const cv::Mat& src) {
                if (src.empty()) return src;
                
                cv::Mat gamma_corrected;
                cv::Mat lookUpTable(1, 256, CV_8U);
                uchar* p = lookUpTable.ptr();
                for( int i = 0; i < 256; ++i)
                    p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 0.7) * 255.0);
                cv::LUT(src, lookUpTable, gamma_corrected);
        
                cv::Mat lab_image; 
                cv::cvtColor(gamma_corrected, lab_image, cv::COLOR_BGR2Lab);
                
                std::vector<cv::Mat> lab_planes(3); 
                cv::split(lab_image, lab_planes);
                
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(lab_planes[0], lab_planes[0]);
                
                cv::merge(lab_planes, lab_image);
                
                cv::Mat dst; 
                cv::cvtColor(lab_image, dst, cv::COLOR_Lab2BGR);
                cv::GaussianBlur(dst, dst, cv::Size(3, 3), 0);
                return dst;
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
        
            void fitNonlinearCenterLine(std::vector<cv::Point2f>& pts, const std_msgs::Header& h, cv::Mat& debug_img, cv::Mat& rgb_img, 
                                        const tf::Transform& trans, const VanishingPointDetector::VPResult& vp_res) {
                
                tf::Transform world_to_cam = trans.inverse();
                struct Point3D { float x, y, z; };
                std::vector<Point3D> local_pts;
                for (const auto& p : pts) {
                    tf::Vector3 p_w(p.x, p.y, 0.0); 
                    tf::Vector3 p_c = world_to_cam * p_w;
                    if (p_c.z() > 0.5 && p_c.z() < 10.0) {
                        local_pts.push_back({(float)p_c.x(), (float)p_c.y(), (float)p_c.z()});
                    }
                }
                std::sort(local_pts.begin(), local_pts.end(), [](const Point3D& a, const Point3D& b) {
                    return a.z < b.z;
                });
        
                std::vector<cv::Point2f> fit_points; 
                if (!local_pts.empty()) {
                    float min_z = local_pts.front().z;
                    float max_z = local_pts.back().z;
                    for (float z = min_z; z <= max_z; z += 0.2) {
                        float sum_x_l = 0, sum_x_r = 0;
                        int n_l = 0, n_r = 0;
                        for (const auto& p : local_pts) {
                            if (std::abs(p.z - z) < 0.15) { 
                                if (p.x < -0.2) { sum_x_l += p.x; n_l++; } 
                                else if (p.x > 0.2) { sum_x_r += p.x; n_r++; } 
                            }
                        }
                        if (n_l > 0 && n_r > 0) {
                            float avg_l = sum_x_l / n_l;
                            float avg_r = sum_x_r / n_r;
                            float center_x = (avg_l + avg_r) / 2.0;
                            fit_points.push_back(cv::Point2f(z, center_x)); 
                            int u = (center_x * config.cam_fx / z) + config.cam_cx;
                            float ground_y_cam = 0.55; 
                            int v = (ground_y_cam * config.cam_fy / z) + config.cam_cy;
                            if (u >= 0 && u < debug_img.cols && v >= 0 && v < debug_img.rows) {
                                cv::circle(debug_img, cv::Point(u, v), 4, cv::Scalar(0, 0, 255), -1); 
                                cv::circle(rgb_img, cv::Point(u, v), 4, cv::Scalar(0, 0, 255), -1); 
                            }
                        }
                    }
                }
        
                if (vp_res.valid) {
                    double delta_u = vp_res.pt.x - config.cam_cx;
                    double slope = delta_u / config.cam_fx;
                    for (float z = 4.0; z <= 12.0; z += 0.5) {
                        float x = z * slope;
                        fit_points.push_back(cv::Point2f(z, x));
                        int u = (x * config.cam_fx / z) + config.cam_cx;
                        float ground_y_cam = 0.55; 
                        int v = (ground_y_cam * config.cam_fy / z) + config.cam_cy;
                        if (u >= 0 && u < debug_img.cols && v >= 0 && v < debug_img.rows) {
                            cv::circle(debug_img, cv::Point(u, v), 4, cv::Scalar(255, 0, 0), -1); 
                            cv::circle(rgb_img, cv::Point(u, v), 4, cv::Scalar(255, 0, 0), -1); 
                        }
                    }
                }
        
                if (fit_points.size() < 5) return;
        
                cv::Mat A(fit_points.size(), 3, CV_32F);
                cv::Mat B(fit_points.size(), 1, CV_32F);
                for(int i=0; i<fit_points.size(); ++i) {
                    float z = fit_points[i].x; float x = fit_points[i].y; 
                    A.at<float>(i,0) = z * z; A.at<float>(i,1) = z; A.at<float>(i,2) = 1.0;
                    B.at<float>(i,0) = x;
                }
        
                cv::Mat coeffs;
                if (!cv::solve(A, B, coeffs, cv::DECOMP_SVD)) return;
        
                float a = coeffs.at<float>(0), b = coeffs.at<float>(1), c = coeffs.at<float>(2);
                smoothed_a_ = 0.8 * smoothed_a_ + 0.2 * a;
                smoothed_b_ = 0.8 * smoothed_b_ + 0.2 * b;
                smoothed_c_ = 0.8 * smoothed_c_ + 0.2 * c;
        
                publishMarkerCurveToGround(smoothed_a_, smoothed_b_, smoothed_c_, trans);
            }
        
            void publishMarkerCurveToGround(float a, float b, float c, const tf::Transform& trans) {
                visualization_msgs::Marker m;
                m.header.frame_id = config.global_frame_id; 
                m.header.stamp = ros::Time::now();
                m.ns = "nav"; m.type = visualization_msgs::Marker::LINE_STRIP;
                m.action = visualization_msgs::Marker::ADD;
                m.id = 0; m.scale.x = 0.1; 
                m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0; 
                m.pose.orientation.w = 1.0;
        
                for (float z = 0.5; z <= 2.0; z += 0.1) {
                    float x_raw = a * z * z + b * z + c; 
                    float x_cam = x_raw;
                    float y_cam = -z;  
                    float z_cam = 0.0;
                    tf::Vector3 p_local(x_cam, y_cam, z_cam); 
                    tf::Vector3 p_map = trans * p_local;
                    geometry_msgs::Point p;
                    p.x = p_map.x();
                    p.y = p_map.y();
                    p.z = 0.1; 
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
        
            void updateAndPublish2DMap() {
                if (pub_2d_map_.getNumSubscribers() == 0) return;
        
                double min_x, min_y, min_z;
                double max_x, max_y, max_z;
                octree_->getMetricMin(min_x, min_y, min_z);
                octree_->getMetricMax(max_x, max_y, max_z);
        
                if (max_x < min_x) {
                    min_x = -10; max_x = 10; min_y = -10; max_y = 10;
                }
        
                nav_msgs::OccupancyGrid map_msg;
                map_msg.header.frame_id = config.global_frame_id;
                map_msg.header.stamp = ros::Time::now();
                
                double res = octree_->getResolution(); 
                map_msg.info.resolution = res;
                
                int width = std::ceil((max_x - min_x) / res) + 20;
                int height = std::ceil((max_y - min_y) / res) + 20;
                map_msg.info.width = width;
                map_msg.info.height = height;
                
                map_msg.info.origin.position.x = min_x - 10 * res;
                map_msg.info.origin.position.y = min_y - 10 * res;
                map_msg.info.origin.position.z = 0.0;
                map_msg.info.origin.orientation.w = 1.0;
        
                map_msg.data.resize(width * height, -1);
        
                for(octomap::ColorOcTree::leaf_iterator it = octree_->begin_leafs(); it != octree_->end_leafs(); ++it) {
                    if (octree_->isNodeOccupied(*it)) {
                        double z = it.getZ();
                        if (z > config.z_ceiling || z < config.lidar_z_min) continue;
        
                        int i = (it.getX() - map_msg.info.origin.position.x) / res;
                        int j = (it.getY() - map_msg.info.origin.position.y) / res;
        
                        if (i >= 0 && i < width && j >= 0 && j < height) {
                            int index = j * width + i;
                            int current_val = map_msg.data[index];
        
                            octomap::OcTreeKey key = it.getKey();
                            float semantic_prob = 0.0;
                            if (semantic_probs_.count(key)) {
                                semantic_prob = semantic_probs_[key];
                            }
        
                            int new_val = 0;
                            if (semantic_prob > config.semantic_thresh) {
                                new_val = 100; 
                            } else if (semantic_prob > 0.4) {
                                new_val = 50;  
                            } else {
                                new_val = 0;   
                            }
        
                            if (new_val > current_val) {
                                map_msg.data[index] = new_val;
                            }
                        }
                    } else {
                        double z = it.getZ();
                        if (z < 0.5 && z > -0.5) {
                            int i = (it.getX() - map_msg.info.origin.position.x) / res;
                            int j = (it.getY() - map_msg.info.origin.position.y) / res;
                            if (i >= 0 && i < width && j >= 0 && j < height) {
                                int index = j * width + i;
                                if (map_msg.data[index] == -1) { 
                                    map_msg.data[index] = 0;
                                }
                            }
                        }
                    }
                }
                pub_2d_map_.publish(map_msg);
            }
        
        private:
            // [NEW] 论文插图专用发布器
            ros::Publisher pub_viz_step1_height_;   // 1. 几何高度分割 (红/绿)
            ros::Publisher pub_viz_step2_hsv_;      // 2. 颜色抑制结果
            ros::Publisher pub_viz_step3_morph_;    // 3. 形态学滤波掩膜
            ros::Publisher pub_viz_step4_lidar_;    // 4. 跨模态投影校验
            ros::Publisher pub_ground_cloud_;

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
            pcl::PointCloud<pcl::PointXYZI>::Ptr latest_high_intensity_cloud_; 
            std::mutex lidar_mutex_;          
            double verify_radius_;            
            double metal_intensity_thresh_;   
            std::shared_mutex map_mutex_; // [修改] 使用读写锁 C++17


            // ================= [🔴 进阶玩法：新增成员变量] =================
            // 1. 全局语义点云地图 (带颜色)
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_semantic_cloud_{new pcl::PointCloud<pcl::PointXYZRGB>()};
            
            // 2. 最新的视觉缓存 (用于跨线程投影)
            cv::Mat latest_cage_mask_;       // 只有黑白二值图：255表示笼子腿，0表示背景
            gtsam::Pose3 latest_cam_pose_;   // 这一帧图像对应的相机位姿 (World -> Camera)
            bool has_new_image_ = false;
            std::mutex viz_mutex_;           // 保护以上变量的锁
            
            // 3. 降采样滤波器 (防止地图点数无限膨胀)
            pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter_;

            // ================= [🔴 针对镂空笼子的特殊数据结构] =================
            struct VoxelKey {
                int x, y, z;
                bool operator==(const VoxelKey& other) const {
                    return x == other.x && y == other.y && z == other.z;
                }
            };

            struct VoxelHash {
                std::size_t operator()(const VoxelKey& k) const {
                    return ((std::size_t)k.x * 73856093) ^ ((std::size_t)k.y * 19349663) ^ ((std::size_t)k.z * 83492791);
                }
            };
            
            // 使用哈希表代替点云直接存储，实现“颜色死锁”逻辑
            // Key: 空间体素索引, Value: 该位置的颜色点
            std::unordered_map<VoxelKey, pcl::PointXYZRGB, VoxelHash> global_voxel_map_;
            float map_resolution_ = 0.05; // 5cm 分辨率
            // ===============================================================
            // ============================================================
            
            StructuralFeatureExtractor feature_extractor_;
            VisualFlowEstimator flow_estimator_;
            ros::Time last_vis_time_; 
        
            struct TemporalInfo {
                int hit_count;            
                int distinct_views;          // [新增] 独立视角计数 (空间多样性)
                gtsam::Point3 last_view_pos; // [新增] 上一次有效观测时的传感器位置
                ros::Time first_seen_time; 
                ros::Time last_seen_time;  
                
                // 初始化构造器
                TemporalInfo() : hit_count(0), distinct_views(0), 
                                 last_view_pos(0,0,0), first_seen_time(0), last_seen_time(0) {}
            };
            
            std::unordered_map<octomap::OcTreeKey, TemporalInfo, octomap::OcTreeKey::KeyHash> temporal_buffer_;
            ros::Time last_clean_time_;
        
            void cleanTemporalBuffer(ros::Time now) {
                if ((now - last_clean_time_).toSec() < 5.0) return; 
                last_clean_time_ = now;
                
                for (auto it = temporal_buffer_.begin(); it != temporal_buffer_.end(); ) {
                    if ((now - it->second.last_seen_time).toSec() > 1.0) {
                        it = temporal_buffer_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        };
        
        int main(int argc, char** argv) {
            ros::init(argc, argv, "semantic_octomap_node");
            SemanticOctomapServer server;
            ros::spin();
            return 0;
        }