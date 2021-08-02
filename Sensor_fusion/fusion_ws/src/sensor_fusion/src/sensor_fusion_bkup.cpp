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

#define ROS_NODE_NAME "sensor_fusion"

float odometry_x = 0;
float odometry_y = 0;
float odometry_yaw = 0;
float servo_angle = 0; // for the inertia element approximation for the servo angle reference vs actual

double imu_x = 0;
double imu_y = 0;
double imu_z = 0;
double imu_vx = 0;
double imu_vy = 0;
double imu_vz = 0;
double imu_yaw = 0;
double imu_roll = 0;
double imu_pitch = 0;

Eigen::Quaternionf imu_q_eig;
Eigen::Vector3f imu_pos = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_vel = Eigen::Vector3f::Zero();
Eigen::Vector3f imu_ang = Eigen::Vector3f::Zero();



float kalman_x = 0;
float kalman_y = 0;
float kalman_x_last = 0;
float kalman_y_last = 0;
float kalman_yaw = 3.14159265358979323846/2;

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


void initEstimatorSystem()
{
  I3x3 = Eigen::MatrixXd::Identity(3,3);
  
  // the larger the L value, the more noise we believe in the initial state
  state_cov_P << 1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0;
                 
//If Q is too high, the KF will tend to use more of the (noisy) measurement. Otherwise KF will put more weight in the process and less in the measurement, then the result might drift away.
  proc_noise_Q << 1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0;
                  
//  If R is high, the filter will respond slowly as it is trusting new measurement less, if R is low, we trust the measurement more, the value might be noisy                 
  meas_noise_R << 1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 5.0;

  best_estimate << kalman_x, kalman_y, kalman_yaw;       

// Initilize the rotation quaternion
  imu_q_eig.vec() = Eigen::Vector3f::Zero();
  imu_q_eig.w() = 1.0f;        
}


void CANCallback( const sensor_msgs::Imu::ConstPtr &msg )
{  
  // Pure IMU 
  double body_ax = - msg -> linear_acceleration.x;
  double body_ay = msg -> linear_acceleration.y;
  double body_az = - msg -> linear_acceleration.z;

  double body_roll_rate = msg -> angular_velocity.y;
  double body_pitch_rate = - msg -> angular_velocity.x;
  double body_yaw_rate = - msg -> angular_velocity.z;


  /********************* Approach 1 ***************************/
  // double imu_ax = body_ax * cos(imu_yaw) - body_ay * sin(imu_yaw);
  // double imu_ay = body_ax * sin(imu_yaw) + body_ay * cos(imu_yaw);

  // imu_yaw += body_yaw_rate * del_t;
  // imu_pitch += body_pitch_rate * del_t;
  // imu_roll += body_roll_rate * del_t;
  /************************************************************/

  /********************* Approach 2 ***************************/
  // Eigen::Matrix<double, 3, 3> rot_yaw;
  // rot_yaw << cos(imu_yaw), -sin(imu_yaw), 0.0,
  //            sin(imu_yaw), cos(imu_yaw), 0.0,
  //            0.0, 0.0, 1.0; 

  // Eigen::Matrix<double, 3, 3> rot_pitch;
  // rot_pitch << 1.0, 0.0, 0.0,
  //            0.0, cos(imu_pitch), -sin(imu_pitch),
  //            0.0, sin(imu_pitch), cos(imu_pitch);

  // Eigen::Matrix<double, 3, 3> rot_roll;
  // rot_roll << cos(imu_roll), 0.0, sin(imu_roll),
  //            0.0, 1.0, 0.0,
  //            -sin(imu_roll), 0.0, cos(imu_roll);


  // Eigen::Matrix<double, 3, 1> body_acc;
  // body_acc << body_ax, body_ay, body_az;
  // Eigen::Matrix<double, 3, 1> global_acc;

  // global_acc = rot_yaw*rot_pitch*rot_roll*body_acc;

  // double imu_ax = global_acc(0);
  // double imu_ay = global_acc(1);

  // imu_yaw += body_yaw_rate * del_t;
  // imu_pitch += body_pitch_rate * del_t;
  // imu_roll += body_roll_rate * del_t;
  /************************************************************/

  /********************* Approach 3 ***************************/
//   tf::Quaternion imu_q, q_rot;
//   imu_q.setRPY(imu_roll, imu_pitch, imu_yaw);
//   imu_q.normalize(); 
//   tf::Matrix3x3 rotation(imu_q);

//   //////// Using quaternion rotation matrices /////////
//   tf::Vector3 body;
//   body.setValue(body_ax, body_ay, body_az);
//   tf::Vector3 global;
//   global = rotation*body;

//   double imu_ax_1 = global.getX();
//   double imu_ay_1 = global.getY();
//   double imu_az_1 = global.getZ();

//   /////////// Using rpy angles ///////////////////
//   // Eigen::Matrix<double, 3, 1> body_acc;
//   // body_acc << body_ax, body_ay, body_az;
//   // Eigen::Matrix<double, 3, 1> global_acc;
//   // Eigen::Matrix<double, 3, 3> rot_roll, rot_pitch, rot_yaw;

//   // rot_yaw << cos(imu_yaw), -sin(imu_yaw), 0.0,
//   //            sin(imu_yaw), cos(imu_yaw), 0.0,
//   //            0.0, 0.0, 1.0; 

//   // rot_pitch << 1.0, 0.0, 0.0,
//   //            0.0, cos(imu_pitch), -sin(imu_pitch),
//   //            0.0, sin(imu_pitch), cos(imu_pitch);

//   // rot_roll << cos(imu_roll), 0.0, sin(imu_roll),
//   //            0.0, 1.0, 0.0,
//   //            -sin(imu_roll), 0.0, cos(imu_roll);

//   // global_acc = rot_yaw*rot_pitch*rot_roll*body_acc;

//   // // double imu_ax = global_acc(0);
//   // // double imu_ay = global_acc(1);
//   // // double imu_az = global_acc(2);

//   double rot_r_1 = body_roll_rate * del_t;
//   double rot_p_1 = body_pitch_rate * del_t;
//   double rot_y_1 = body_yaw_rate * del_t;

//   q_rot.setRPY(rot_r_1, rot_p_1, rot_y_1);
//   q_rot.normalize();
//   imu_q = q_rot * imu_q; //////////////////////////////// What is the right order? 
//   imu_q.normalize();

//   tf::Matrix3x3 new_angle(imu_q);
//   new_angle.getRPY(imu_roll, imu_pitch, imu_yaw);

// /************************************************************/

//   /////////// Using Eigen ///////////////////
//   Eigen::Vector3f global_acc, body_acc; ///< Position

//   body_acc(0) = body_ax;
//   body_acc(1) = body_ay;
//   body_acc(2) = body_az;

//   Eigen::Matrix3f R = imu_q_eig.toRotationMatrix();

//   global_acc = R * body_acc;

//   double imu_ax = global_acc(0);
//   double imu_ay = global_acc(1);

//   double rot_r = body_roll_rate * del_t;
//   double rot_p = body_pitch_rate * del_t;
//   double rot_y = body_yaw_rate * del_t;

//   Eigen::Vector3f delta_rot;
//   delta_rot(0) =  rot_r;
//   delta_rot(1) =  rot_p;
//   delta_rot(2) =  rot_y;

//   Eigen::Quaternionf q_rot_eig; ///< Quaternion
//   // makeQuaternionFromVector( delta_rot, q_rot);
//   float phi = delta_rot.norm();
//   Eigen::Vector3f u = delta_rot / phi; // u is a unit vector
//   q_rot_eig.vec() = u * sin( phi / 2.0 );
//   q_rot_eig.w()   =     cos( phi / 2.0 );

//   imu_q_eig =  imu_q_eig * (q_rot_eig);   //////////////////////////////// Right order? 

//   std::cout << "DIFF: " << imu_ax - imu_ax_1 << " " << imu_ay - imu_ay_1 << std::endl;
  
//   //////////////////////////////////////////////


//   imu_x += imu_vx * del_t + 0.5 * imu_ax * del_t * del_t;
//   imu_y += imu_vy * del_t + 0.5 * imu_ay * del_t * del_t;

//   imu_vx += imu_ax * del_t;
//   imu_vy += imu_ay * del_t;

//   nav_msgs::Odometry pure_imu;
//   pure_imu.header.frame_id = "map";
//   pure_imu.pose.pose.position.x = imu_x;
//   pure_imu.pose.pose.position.y = imu_y;
  
//   geometry_msgs::Quaternion q;
//   // q = tf::createQuaternionMsgFromRollPitchYaw(imu_roll, imu_pitch, imu_yaw);
//   // pure_imu.pose.pose.orientation = q;

//   pure_imu.pose.pose.orientation.w = imu_q_eig.w();
//   pure_imu.pose.pose.orientation.x = imu_q_eig.x();
//   pure_imu.pose.pose.orientation.y = imu_q_eig.y();
//   pure_imu.pose.pose.orientation.z = imu_q_eig.z();
  
//   imu_pos_pub_.publish(pure_imu);

  double imu_ax, imu_ay, imu_az, imu_vx, imu_vy, imu_vz, imu_x, imu_y, imu_z;


  Eigen::Vector3f imu_lin_acc, imu_ang_vel;
  imu_lin_acc(0) = msg -> linear_acceleration.x;
  imu_lin_acc(1) = msg -> linear_acceleration.y;
  imu_lin_acc(2) = msg -> linear_acceleration.z;
  imu_ang_vel(0) = msg -> angular_velocity.x;
  imu_ang_vel(1) = msg -> angular_velocity.y;
  imu_ang_vel(2) = msg -> angular_velocity.z;
  
  Eigen::Vector3f gravity = Eigen::Vector3f::Zero();

  Eigen::Matrix3f R = imu_q_eig.toRotationMatrix();

  Eigen::Vector3f acc_projection = R*imu_lin_acc + gravity;

  acc_projection(2) = 0.0;

  imu_pos += imu_vel*del_t + 0.5*acc_projection*del_t*del_t;
  imu_vel += acc_projection*del_t;



  Eigen::Vector3f imu_ang = imu_ang_vel * del_t;

  Eigen::Quaternionf q_rot_eig; ///< Quaternion
  float phi = imu_ang.norm();
  Eigen::Vector3f u = imu_ang / phi; // u is a unit vector
  q_rot_eig.vec() = u * sin( phi / 2.0 );
  q_rot_eig.w()   =     cos( phi / 2.0 );

  imu_q_eig =  imu_q_eig * (q_rot_eig); 

  nav_msgs::Odometry pure_imu;
  pure_imu.header.frame_id = "map";
  pure_imu.pose.pose.position.x = - imu_pos(0);
  pure_imu.pose.pose.position.y = imu_pos(1);
  
  geometry_msgs::Quaternion q;

  Eigen::Vector3f imu_2_car;
  imu_2_car(0) = 0.0; 
  imu_2_car(1) = 3.1415926;  //3.1415926;
  imu_2_car(2) = 0.0;

  Eigen::Quaternionf q_imu_car; ///< Quaternion
  phi = imu_2_car.norm();
  u = imu_2_car / phi; // u is a unit vector
  q_imu_car.vec() = u * sin( phi / 2.0 );
  q_imu_car.w()   =     cos( phi / 2.0 );

  Eigen::Quaternionf q_car = imu_q_eig*q_imu_car;

  pure_imu.pose.pose.orientation.w = q_car.w();
  pure_imu.pose.pose.orientation.x = q_car.x();
  pure_imu.pose.pose.orientation.y = q_car.y();
  pure_imu.pose.pose.orientation.z = q_car.z();
  
  imu_pos_pub_.publish(pure_imu);




  // Pure odometry
  float car_speed = msg -> orientation.x;
  float servo_angle_raw = msg -> orientation.y;
  // servo_angle_raw =  - 26 * sin((servo_angle_raw - 17500)*3.1415926/180/68.553) * 3.1415926/180;
  // servo_angle_raw =   (servo_angle_raw - 17500)*7.56e-5;
  float filter_param = 0.01;

  servo_angle = (1 - filter_param) * servo_angle + filter_param * servo_angle_raw;
  //servo_angle = 0.62 * servo_angle_raw;  // 0.867

  float wheel_base = 0.257; // in meters

  odometry_x -= car_speed * sin(odometry_yaw) * del_t;
  odometry_y += car_speed * cos(odometry_yaw) * del_t;
  
  odometry_yaw += car_speed * tan(servo_angle) * del_t / wheel_base;

  // std::cout << "ODOM POSE: " << odometry_x << "," << odometry_y << ", " << odometry_yaw << std::endl;
  
  nav_msgs::Odometry pure_odometry;
  pure_odometry.header.frame_id = "map";
  pure_odometry.pose.pose.position.x = odometry_x;
  pure_odometry.pose.pose.position.y = odometry_y;
  pure_odometry.pose.pose.position.z = 0.0;

  q = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, odometry_yaw);
  pure_odometry.pose.pose.orientation = q;

  // Just for visualize
  pure_odometry.twist.twist.linear.x = servo_angle_raw * 180 / 3.1415926;
  pure_odometry.twist.twist.linear.y = servo_angle * 180 / 3.1415926;

  odometry_pos_pub_.publish(pure_odometry);

  // EKF
  // step 1: prediction
  sys_A << 0.0, 0.0, -car_speed * sin(kalman_yaw),
           0.0, 0.0, car_speed * cos(kalman_yaw),
           0.0, 0.0, 0.0;

  sys_A_t = sys_A.transpose();

  /************************************************************/
  // propogate states using imu, and use the results as measurements
  imu_ax = body_ax * sin(kalman_yaw) + body_ay * cos(kalman_yaw);
  imu_ay = - body_ax * cos(kalman_yaw) + body_ay * sin(kalman_yaw);
  float imu_vx_est = (kalman_x - kalman_x_last) / del_t;
  float imu_vy_est = (kalman_y - kalman_y_last) / del_t;
  float imu_x_est = kalman_x + imu_vx_est * del_t + 0.5 * imu_ax * del_t * del_t;
  float imu_y_est = kalman_y + imu_vy_est * del_t + 0.5 * imu_ay * del_t * del_t;
  float imu_yaw_est = kalman_yaw + body_yaw_rate * del_t;
  kalman_x_last = kalman_x;
  kalman_y_last = kalman_y;
  /************************************************************/

  // The real prediction step using odometry      
  kalman_x += car_speed * cos(kalman_yaw) * del_t;
  kalman_y += car_speed * sin(kalman_yaw) * del_t;
  kalman_yaw += car_speed * tan(servo_angle) * del_t / wheel_base;

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


  nav_msgs::Odometry ekf;
  ekf.header.frame_id = "map";
  ekf.pose.pose.position.x = kalman_x;
  ekf.pose.pose.position.y = kalman_y;
  ekf.pose.pose.position.z = 0.0;

  q = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, kalman_yaw);
  ekf.pose.pose.orientation = q;
  
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

