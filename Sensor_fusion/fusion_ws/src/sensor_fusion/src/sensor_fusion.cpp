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

float odometry_x = 0;
float odometry_y = 0;
float odometry_yaw = 0;
float servo_angle = 0; // for the inertia element approximation for the servo angle reference vs actual

double imu_x = 0;
double imu_y = 0;
double imu_z = 0;
Eigen::Quaternionf imu_q;
Eigen::Vector3f imu_pos = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_vel = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_ang = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_lin_acc_last = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_ang_vel_last = Eigen::Vector3f::Zero();

Eigen::Quaternionf kalman_q;
Eigen::Vector3f kalman_pos = Eigen::Vector3f::Zero();
Eigen::Vector3f kalman_vel = Eigen::Vector3f::Zero();
Eigen::Vector3f kalman_ang = Eigen::Vector3f::Zero();
Eigen::Vector3f kalman_lin_acc_last = Eigen::Vector3f::Zero();
Eigen::Vector3f kalman_ang_vel_last = Eigen::Vector3f::Zero();

float kalman_x = 0;
float kalman_y = 0;
float kalman_x_last = 0;
float kalman_y_last = 0;
float kalman_roll = 0;
float kalman_pitch = 0;
float kalman_yaw = 0;
float kalman_yaw_last = 0;
float servo_angle_kf = 0;

float del_t = 0.01;

ros::Publisher imu_pos_pub_;
ros::Publisher odometry_pos_pub_;
ros::Publisher kalman_pos_pub_;

const int nStates = 3;
const int nCtrl = 2;
const int nMeas = 3;

Eigen::Matrix<double, nStates, nStates> state_cov_P;
Eigen::Matrix<double, nStates, nStates> proc_noise_Q;
Eigen::Matrix<double, nMeas, nMeas> meas_noise_R;
Eigen::Matrix<double, nStates, nMeas> kalman_gain_k;
Eigen::Matrix<double, nStates, nStates> sys_A, sys_A_t;

Eigen::Matrix<double, nStates, 1> best_estimate;
Eigen::Matrix<double, nMeas, 1> measurement;

Eigen::Matrix<double, 3, 3> I3x3;

Eigen::MatrixXd innov;

void makeQuaternionFromVector( Eigen::Vector3f& inVec, Eigen::Quaternionf& outQuat )
{
    float phi = inVec.norm();
    Eigen::Vector3f u = inVec / phi; // u is a unit vector

    outQuat.vec() = u * sin( phi / 2.0 );
    outQuat.w()   =     cos( phi / 2.0 );
}

void initEstimatorSystem()
{
  I3x3 = Eigen::MatrixXd::Identity(3,3);
  
  // the larger the L value, the more noise we believe in the initial state
  state_cov_P << 1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0;
                 
// If Q is too high, the KF will tend to use more of the (noisy) measurement. Otherwise KF will put more weight in the process and less in the measurement, then the result might drift away.
// Use IMU as prediction step. The x_y prediction is noisy, so put large number there. 
// Seems that large numbers mean we trust IMU more. 
  proc_noise_Q << 1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 100.0;

  // proc_noise_Q << 0.0, 0.0, 0.0,
  //                 0.0, 0.0, 0.0,
  //                 0.0, 0.0, 0.0;
                  
// If R is high, the filter will respond slowly as it is trusting new measurement less, if R is low, we trust the measurement more, the value might be noisy                 
// Use odometry as measurement step.   
  meas_noise_R << 100.0, 0.0, 0.0,
                  0.0, 100.0, 0.0,
                  0.0, 0.0, 1.0;

  // meas_noise_R << 0.0, 0.0, 0.0,
  //                 0.0, 0.0, 0.0,
  //                 0.0, 0.0, 0.0;

  best_estimate << kalman_x, kalman_y, kalman_yaw;       

// Initilize the rotation quaternion
  imu_q = Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());     
  kalman_q = Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());     
}

void CANCallback( const sensor_msgs::Imu::ConstPtr &msg )
{  
  /******************* Pure IMU **************************/
  float imu_filter_param = 1.0;
  Eigen::Vector3f imu_lin_acc, imu_ang_vel;
  imu_lin_acc(0) = -msg -> linear_acceleration.x;
  imu_lin_acc(1) = msg -> linear_acceleration.y;
  imu_lin_acc(2) = -msg -> linear_acceleration.z;
  imu_ang_vel(0) = -msg -> angular_velocity.x;
  imu_ang_vel(1) = msg -> angular_velocity.y;
  imu_ang_vel(2) = -msg -> angular_velocity.z;
  
  Eigen::Vector3f gravity = Eigen::Vector3f::Zero();
  Eigen::Matrix3f R = imu_q.toRotationMatrix();
  Eigen::Vector3f acc_proj_last = R*imu_lin_acc_last + gravity;
  Eigen::Vector3f acc_projection = R*imu_lin_acc + gravity;
  acc_projection = imu_filter_param * acc_projection + (1 - imu_filter_param) * acc_proj_last;
  acc_projection(2) = 0.0; // force the z axis acceleration to 0

  imu_pos += imu_vel*del_t + 0.5*acc_projection*del_t*del_t;
  imu_vel += acc_projection*del_t;

  Eigen::Quaternionf q_rot;
  Eigen::Vector3f imu_ang = (imu_filter_param * imu_ang_vel + (1 - imu_filter_param) * imu_ang_vel_last) * del_t;
  // makeQuaternionFromVector(imu_ang, q_rot);
  q_rot = Eigen::AngleAxisf(imu_ang(0), Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(imu_ang(1), Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(imu_ang(2), Eigen::Vector3f::UnitZ());
  imu_q =  imu_q * (q_rot); 

  auto euler = imu_q.toRotationMatrix().eulerAngles(0, 1, 2);
  // std::cout << "Euler from quaternion in roll, pitch, yaw"<< std::endl << euler << std::endl;

  imu_lin_acc_last = imu_lin_acc;
  imu_ang_vel_last = imu_ang_vel;

  nav_msgs::Odometry pure_imu;
  pure_imu.header.frame_id = "map";
  pure_imu.pose.pose.position.x = - imu_pos(0);
  pure_imu.pose.pose.position.y = imu_pos(1);
  
  pure_imu.pose.pose.orientation.w = imu_q.w();
  pure_imu.pose.pose.orientation.x = imu_q.x();
  pure_imu.pose.pose.orientation.y = imu_q.y();
  pure_imu.pose.pose.orientation.z = imu_q.z();
  
  imu_pos_pub_.publish(pure_imu);



  /******************* Pure Odometry **************************/
  float car_speed = msg -> orientation.x;
  float servo_angle_raw = msg -> orientation.y;
  float filter_param = 0.01;

  servo_angle = (1 - filter_param) * servo_angle + filter_param * servo_angle_raw;
  //servo_angle = 0.62 * servo_angle_raw;  // 0.867

  float wheel_base = 0.257; // in meters

  odometry_x -= car_speed * sin(odometry_yaw) * del_t;
  odometry_y += car_speed * cos(odometry_yaw) * del_t;
  
  odometry_yaw += car_speed * tan(servo_angle) * del_t / wheel_base;
  
  nav_msgs::Odometry pure_odometry;
  pure_odometry.header.frame_id = "map";
  pure_odometry.pose.pose.position.x = odometry_x;
  pure_odometry.pose.pose.position.y = odometry_y;
  pure_odometry.pose.pose.position.z = 0.0;
  
  geometry_msgs::Quaternion q;
  q = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, odometry_yaw);
  pure_odometry.pose.pose.orientation = q;

  // Just for visualize
  pure_odometry.twist.twist.linear.x = servo_angle_raw * 180 / 3.1415926;
  pure_odometry.twist.twist.linear.y = servo_angle * 180 / 3.1415926;

  odometry_pos_pub_.publish(pure_odometry);



  /******************* EKF **************************/
  // step 1: prediction
  // propogate states using imu, and use the results as measurements

  float param_imu = 1.0;
  Eigen::Vector3f imu_lin_acc_kf, imu_ang_vel_kf;
  imu_lin_acc_kf(0) = -msg -> linear_acceleration.x;
  imu_lin_acc_kf(1) = msg -> linear_acceleration.y;
  imu_lin_acc_kf(2) = -msg -> linear_acceleration.z;
  imu_ang_vel_kf(0) = -msg -> angular_velocity.x;
  imu_ang_vel_kf(1) = msg -> angular_velocity.y;
  imu_ang_vel_kf(2) = -msg -> angular_velocity.z;
  
  Eigen::Matrix3f R_kf = kalman_q.toRotationMatrix();
  Eigen::Vector3f acc_proj_last_kf = R_kf*kalman_lin_acc_last;
  Eigen::Vector3f acc_projection_kf = R_kf*imu_lin_acc_kf;
  acc_projection_kf = param_imu * acc_projection_kf + (1 - param_imu) * acc_proj_last_kf;

  float imu_vx_est = (kalman_x - kalman_x_last) / del_t;
  float imu_vy_est = (kalman_y - kalman_y_last) / del_t;
  float imu_x_est = kalman_x + imu_vx_est * del_t + 0.5 * (-acc_projection_kf(0)) * del_t * del_t;
  float imu_y_est = kalman_y + imu_vy_est * del_t + 0.5 * acc_projection_kf(0) * del_t * del_t;


  Eigen::Quaternionf q_rot_kf;
  Eigen::Vector3f imu_ang_kf = (param_imu * imu_ang_vel_kf + (1 - param_imu) * kalman_ang_vel_last) * del_t;
  float imu_yaw_est = kalman_yaw + imu_ang_kf(2);

  kalman_x_last = kalman_x;
  kalman_y_last = kalman_y;

  kalman_lin_acc_last = imu_lin_acc_kf;
  kalman_ang_vel_last = imu_ang_vel_kf;

  float car_speed_kf = msg -> orientation.x;
  float servo_angle_raw_kf = msg -> orientation.y;
  float param_odom = 0.01;

  sys_A << 0.0, 0.0, -car_speed_kf * cos(kalman_yaw),
           0.0, 0.0, -car_speed_kf * sin(kalman_yaw),
           0.0, 0.0, 0.0;

  sys_A_t = sys_A.transpose();

  servo_angle_kf = (1 - param_odom) * servo_angle_kf + param_odom * servo_angle_raw_kf;
  //servo_angle_kf = 0.62 * servo_angle_raw_kf;  // 0.867

  float wheel_base_kf = 0.257; // in meters

  kalman_x -= car_speed_kf * sin(kalman_yaw) * del_t;
  kalman_y += car_speed_kf * cos(kalman_yaw) * del_t;
  kalman_yaw += car_speed_kf * tan(servo_angle_kf) * del_t / wheel_base_kf;

  state_cov_P = sys_A*state_cov_P*sys_A_t + proc_noise_Q;


  // step 2: updation
  innov = ( state_cov_P + meas_noise_R ).inverse();
  kalman_gain_k = state_cov_P * innov;
  best_estimate << kalman_x, kalman_y, kalman_yaw;
  measurement << imu_x_est, imu_y_est, imu_yaw_est;
  best_estimate = best_estimate + kalman_gain_k * (measurement - best_estimate);
  state_cov_P = (I3x3 - kalman_gain_k) * state_cov_P;
  kalman_x = best_estimate(0);
  kalman_y = best_estimate(1);
  kalman_yaw = best_estimate(2);

  float yaw_diff = kalman_yaw - kalman_yaw_last;
  q_rot_kf = Eigen::AngleAxisf(imu_ang_kf(0), Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(imu_ang_kf(1), Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(yaw_diff, Eigen::Vector3f::UnitZ());
  kalman_q =  kalman_q * (q_rot_kf); 
  kalman_yaw_last = kalman_yaw;

  nav_msgs::Odometry ekf;
  ekf.header.frame_id = "map";
  ekf.pose.pose.position.x = kalman_x;
  ekf.pose.pose.position.y = kalman_y;
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

