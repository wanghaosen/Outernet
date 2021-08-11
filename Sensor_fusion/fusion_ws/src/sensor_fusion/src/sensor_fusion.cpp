#include <ros/ros.h>
#include <iostream>
#include <mutex>
#include <chrono>
#include <tf/tf.h>
#include <eigen3/Eigen/Dense>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <ecl/geometry/angle.hpp>

#define ROS_NODE_NAME "sensor_fusion"

/********* Global Variables for Pure Odometry **********/
float odometry_x = 0;
float odometry_y = 0;
float odometry_yaw = 0;
float servo_angle = 0; // for the inertia element approximation for the servo angle reference vs actual

/********* Global Variables for Pure IMU **********/
Eigen::Quaternionf imu_q;
Eigen::Vector3f imu_pos = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_vel = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_ang = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_lin_acc_last = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_ang_vel_last = Eigen::Vector3f::Zero();

/********* Global Variables for EKF **********/
Eigen::Quaternionf kalman_q;
float kalman_yaw_last = 0; 

float kalman_x = 0;
float kalman_y = 0;
float kalman_yaw = 0;
float kalman_vx = 0;
float kalman_vy = 0;
float kalman_vyaw = 0;
float kalman_steering = 0;

float del_t = 0.01;
const int nStates = 7;
const int nCtrl = 2;
const int nMeas = 7;
Eigen::Matrix<double, nStates, nStates> state_cov_P;
Eigen::Matrix<double, nStates, nStates> proc_noise_Q;
Eigen::Matrix<double, nMeas, nMeas> meas_noise_R;
Eigen::Matrix<double, nStates, nMeas> kalman_gain_k;
Eigen::Matrix<double, nStates, nStates> sys_A, sys_A_t;
Eigen::Matrix<double, nStates, 1> best_estimate;
Eigen::Matrix<double, nMeas, 1> measurement;
Eigen::Matrix<double, 7, 7> I7x7;
Eigen::MatrixXd innov;

ros::Publisher imu_pos_pub_;
ros::Publisher odometry_pos_pub_;
ros::Publisher kalman_pos_pub_;

float wheel_base = 0.257; // in meters


void initEstimatorSystem()
{
  // Linear system model based on IMU equations
  sys_A << 1.0, 0.0, 0.0, del_t, 0.0, 0.0, 0.0,
           0.0, 1.0, 0.0, 0.0, del_t, 0.0, 0.0,
           0.0, 0.0, 1.0, 0.0, 0.0, del_t, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  sys_A_t = sys_A.transpose();
  
  
  I7x7 = Eigen::MatrixXd::Identity(7,7);
  
  // the larger the L value, the more noise we believe in the initial state
  state_cov_P << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
                 
// If Q is too high, the KF will tend to use more of the (noisy) measurement. Otherwise KF will put more weight in the process and less in the measurement, then the result might drift away.
// Use IMU as prediction step. The x_y prediction is noisy, so put large number there. 
  proc_noise_Q << 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 
                  0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0;   // candidate1: 100.0  // candidate2: 10

                  // One tricky param is the steering. See how much we trust the servo and the acc estimation.
                  
// If R is high, the filter will respond slowly as it is trusting new measurement less, if R is low, we trust the measurement more, the value might be noisy                 
// Use odometry as measurement step.   
  meas_noise_R << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 175.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                  0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;   // 175


  best_estimate << kalman_x, kalman_y, kalman_yaw, kalman_vx, kalman_vy, kalman_vyaw, kalman_steering;       

// Initilize the rotation quaternion
  imu_q = Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());     
  kalman_q = Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());     
}

void CANCallback( const sensor_msgs::Imu::ConstPtr &msg )
{ 
  /*************** Read raw data from msg ****************/ 
  Eigen::Vector3f imu_lin_acc, imu_ang_vel;
  imu_lin_acc(0) = -msg -> linear_acceleration.x;
  imu_lin_acc(1) = msg -> linear_acceleration.y;
  imu_lin_acc(2) = -msg -> linear_acceleration.z;
  imu_ang_vel(0) = -msg -> angular_velocity.x;
  imu_ang_vel(1) = msg -> angular_velocity.y;
  imu_ang_vel(2) = -msg -> angular_velocity.z;

  float car_speed = msg -> orientation.x;
  float servo_angle_raw = msg -> orientation.y;  // + 0.1;

  float imu_rc_filter_param = 0.5; // 0.5
  float servo_rc_filter_param = 0.05;  // candidate1: 0.01;   // candidate2: 0.05
  float beta, center_speed;
  
  // /******************* Pure IMU **************************/
  // Eigen::Vector3f gravity = Eigen::Vector3f::Zero();
  // Eigen::Matrix3f R = imu_q.toRotationMatrix();
  // Eigen::Vector3f acc_proj_last = R*imu_lin_acc_last + gravity;
  // Eigen::Vector3f acc_projection = R*imu_lin_acc + gravity;
  // acc_projection = imu_rc_filter_param * acc_projection + (1 - imu_rc_filter_param) * acc_proj_last;

  // imu_pos += imu_vel*del_t + 0.5*acc_projection*del_t*del_t;
  // imu_vel += acc_projection*del_t;

  // Eigen::Vector3f imu_ang = (imu_rc_filter_param * imu_ang_vel + (1 - imu_rc_filter_param) * imu_ang_vel_last) * del_t;
  // Eigen::Quaternionf q_rot = Eigen::AngleAxisf(imu_ang(0), Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(imu_ang(1), Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(imu_ang(2), Eigen::Vector3f::UnitZ());
  // imu_q =  imu_q * (q_rot); 

  // // auto euler = imu_q.toRotationMatrix().eulerAngles(0, 1, 2);

  // nav_msgs::Odometry pure_imu;
  // pure_imu.header.frame_id = "map";
  // pure_imu.pose.pose.position.x = 0; //imu_pos(0);
  // pure_imu.pose.pose.position.y = 0; //imu_pos(1);
  
  // pure_imu.pose.pose.orientation.w = imu_q.w();
  // pure_imu.pose.pose.orientation.x = imu_q.x();
  // pure_imu.pose.pose.orientation.y = imu_q.y();
  // pure_imu.pose.pose.orientation.z = imu_q.z();
  
  // imu_pos_pub_.publish(pure_imu);


  // /******************* Pure Odometry **************************/
  
  // servo_angle = servo_rc_filter_param * servo_angle_raw + (1 - servo_rc_filter_param) * servo_angle;
  // // servo_angle = 0.62 * servo_angle_raw;  // 0.867

  // beta = atan2(tan(servo_angle), 2);
  // center_speed = car_speed/cos(beta);

  // odometry_x -= center_speed * sin(odometry_yaw + beta) * del_t;
  // odometry_y += center_speed * cos(odometry_yaw + beta) * del_t;
  // odometry_yaw += car_speed * tan(servo_angle) * del_t / wheel_base;
  
  // nav_msgs::Odometry pure_odometry;
  // pure_odometry.header.frame_id = "map";
  // pure_odometry.pose.pose.position.x = odometry_x;
  // pure_odometry.pose.pose.position.y = odometry_y;
  // pure_odometry.pose.pose.position.z = 0.0;
  
  // geometry_msgs::Quaternion q;
  // q = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, odometry_yaw);
  // pure_odometry.pose.pose.orientation = q;

  // // // Just for visualizing in bag
  // // pure_odometry.twist.twist.linear.x = servo_angle_raw * 180 / 3.1415926;
  // // pure_odometry.twist.twist.linear.y = servo_angle * 180 / 3.1415926;

  // odometry_pos_pub_.publish(pure_odometry);


  /******************* EKF **************************/
  // step 1: prediction

  // propogate states using imu
  Eigen::Matrix3f R_kf = kalman_q.toRotationMatrix();
  Eigen::Vector3f acc_proj_last_kf = R_kf*imu_lin_acc_last;
  Eigen::Vector3f acc_projection_kf = R_kf*imu_lin_acc;
  acc_projection_kf = imu_rc_filter_param * acc_projection_kf + (1 - imu_rc_filter_param) * acc_proj_last_kf;
  Eigen::Vector3f imu_ang_vel_kf = imu_rc_filter_param * imu_ang_vel + (1 - imu_rc_filter_param) * imu_ang_vel_last;
  // Eigen::Vector3f imu_ang_vel_kf = imu_ang_vel;
  
  imu_lin_acc_last = imu_lin_acc;
  imu_ang_vel_last = imu_ang_vel;

  // propagate state forward
  float imu_vx_est = kalman_vx + acc_projection_kf(0) * del_t;
  float imu_vy_est = kalman_vy + acc_projection_kf(1) * del_t;

  float yaw_pi_2_pi = ecl::wrap_angle(kalman_yaw);
  float imu_steering_est = atan2(imu_vy_est, imu_vx_est) - yaw_pi_2_pi - 3.1415926/2;
  // float imu_steering_est = atan2(kalman_vy, kalman_vx) - yaw_pi_2_pi - 3.1415926/2;
  imu_steering_est = ecl::wrap_angle(imu_steering_est);
  if(imu_steering_est > (26*3.14/180)){imu_steering_est = 26*3.14/180;}
  if(imu_steering_est < (-26*3.14/180)) {imu_steering_est = - 26*3.14/180;}
  
  float imu_vyaw_est = imu_ang_vel_kf(2);
  float imu_x_est = kalman_x + kalman_vx * del_t + 0.5 * acc_projection_kf(0) * del_t * del_t;
  float imu_y_est = kalman_y + kalman_vy * del_t + 0.5 * acc_projection_kf(1) * del_t * del_t;
  float imu_yaw_est = kalman_yaw + imu_vyaw_est * del_t;

  // auto euler = R_kf.eulerAngles(0, 1, 2);
  // std::cout << "At yaw angle " << kalman_yaw << ": " << std::endl;
  // std::cout << "Pitch: " << euler(0) << " Roll: " << euler(1) << " Yaw: " << euler(2) << std::endl;

  // update the covariance matrix
  state_cov_P = sys_A*state_cov_P*sys_A_t + proc_noise_Q;


  // step 2: updation

  // calculate the pseudo-measurement based on wheel odometry
  float odom_steering_est = servo_rc_filter_param * servo_angle_raw + (1 - servo_rc_filter_param) * kalman_steering;
  beta = atan2(tan(kalman_steering), 2);
  center_speed = car_speed/cos(beta);

  if(odom_steering_est > (26*3.14/180)){odom_steering_est = 26*3.14/180;}
  if(odom_steering_est < (-26*3.14/180)) {odom_steering_est = - 26*3.14/180;}

  float odom_vx_est = -center_speed * sin(kalman_yaw + beta);
  float odom_vy_est = center_speed * cos(kalman_yaw + beta);
  float odom_vyaw_est = car_speed * tan(kalman_steering) / wheel_base;
  // float odom_vyaw_est = car_speed * tan(odom_steering_est) / wheel_base;
  float odom_x_est = kalman_x + odom_vx_est * del_t;
  float odom_y_est = kalman_y + odom_vy_est * del_t;
  float odom_yaw_est = kalman_yaw + odom_vyaw_est * del_t;

  std::cout << "At yaw angle " << kalman_yaw << ": " << std::endl;
  std::cout << "STEERING: " << "  IMU: " << imu_steering_est << " ODOM: " << odom_steering_est << " EKF: " << kalman_steering << std::endl;
  std::cout << "Yaw Speed: " << "  IMU: " << imu_vyaw_est << " ODOM: " << odom_vyaw_est << " EKF: " << kalman_vyaw << std::endl;
  std::cout << "Velocity: " << "  IMU: " << imu_vx_est << " , " << imu_vy_est << " ODOM: " << odom_vx_est << " , " << odom_vy_est << " EKF: " << kalman_vx << " , " << kalman_vy << std::endl << std::endl;


  // update parameter matrices, calculate the state estimation
  innov = ( state_cov_P + meas_noise_R ).inverse();
  kalman_gain_k = state_cov_P * innov;
  best_estimate << imu_x_est, imu_y_est, imu_yaw_est, imu_vx_est, imu_vy_est, imu_vyaw_est, imu_steering_est;
  measurement << odom_x_est, odom_y_est, odom_yaw_est, odom_vx_est, odom_vy_est, odom_vyaw_est, odom_steering_est;
  best_estimate = best_estimate + kalman_gain_k * (measurement - best_estimate);
  state_cov_P = (I7x7 - kalman_gain_k) * state_cov_P;

  kalman_x = best_estimate(0);
  kalman_y = best_estimate(1);
  kalman_yaw = best_estimate(2);
  kalman_vx = best_estimate(3);
  kalman_vy = best_estimate(4);
  kalman_vyaw = best_estimate(5);
  kalman_steering = best_estimate(6);
  

  float yaw_diff = kalman_yaw - kalman_yaw_last;
  Eigen::Quaternionf q_rot_kf = Eigen::AngleAxisf((imu_ang_vel_kf(0) * del_t), Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf((imu_ang_vel_kf(1) * del_t), Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(yaw_diff, Eigen::Vector3f::UnitZ());
  kalman_q =  kalman_q * (q_rot_kf); 
  kalman_yaw_last = kalman_yaw;


  nav_msgs::Odometry ekf;
  ekf.header.frame_id = "map";
  ekf.pose.pose.position.x = kalman_x;  // imu_x_est; // odom_x_est; // kalman_x;
  ekf.pose.pose.position.y = kalman_y;  // imu_y_est; // odom_y_est; // kalman_y;
  ekf.pose.pose.position.z = 0;
 
  ekf.pose.pose.orientation.w = kalman_q.w();
  ekf.pose.pose.orientation.x = kalman_q.x();
  ekf.pose.pose.orientation.y = kalman_q.y();
  ekf.pose.pose.orientation.z = kalman_q.z();
  
  kalman_pos_pub_.publish(ekf);
  
}

int main( int argc, char **argv )
{
  ros::init( argc, argv, ROS_NODE_NAME );
  
  ros::NodeHandle nh_;

  initEstimatorSystem();

  ros::Subscriber CAN_sub_;
  CAN_sub_ = nh_.subscribe( "/CAN_SIGNALS", 1, CANCallback );

  imu_pos_pub_ = nh_.advertise <nav_msgs::Odometry> ( "/imu_pos", 1, true );
  odometry_pos_pub_ = nh_.advertise <nav_msgs::Odometry> ( "/odometry_pos", 1, true );
  kalman_pos_pub_ = nh_.advertise <nav_msgs::Odometry> ( "/kalman_pos", 1, true );

  ros::spin();
  
  return 0;
}

