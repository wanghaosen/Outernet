### Convert recorded csv file data to rosbag


import pandas as pd
df = pd.read_csv('input.csv')

import rospy
import rosbag
import math
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped

with rosbag.Bag('output.bag', 'w') as bag:
    rospy.init_node('csv2bag',anonymous=True)
    now = rospy.Time.now()
    for row in range(df.shape[0]):
        now = now + rospy.Duration(0.01)

        # IMU: accelerations are in m/s^2, rotation velocities are in rad/sec
        imu_msg = Imu()
        imu_msg.header.stamp = now

        imu_msg.angular_velocity.x = df['gyro_x'][row] * 0.00053263221
        imu_msg.angular_velocity.y = df['gyro_y'][row] * 0.00053263221
        imu_msg.angular_velocity.z = df['gyro_z'][row] * 0.00053263221

        imu_msg.linear_acceleration.x = df['acc_x'][row] * 0.00239257812
        imu_msg.linear_acceleration.y = df['acc_y'][row] * 0.00239257812
        imu_msg.linear_acceleration.z = df['acc_z'][row] * 0.00239257812

        imu_msg.orientation.x = df['motor'][row] * 0.00024868693 * 3.1415926
        #imu_msg.orientation.y = (df['servo'][row] - 17500) * 0.00008726646 + 0.096 
        #imu_msg.orientation.y = 26 * 3.1415926/180 * math.sin((df['servo'][row] - 17500) / 68.553)
        #imu_msg.orientation.y = 27.446 * math.sin((df['servo'][row] - 17500) * 3.1415926 / 180 / 84.13) * 3.1415926 / 180 + 0.105
        #imu_msg.orientation.z = (df['servo'][row] - 17500) * 26 * 3.1415926 / 180 / 6000 + 0.077
        imu_msg.orientation.y = 27.446 * math.sin((df['servo'][row] - 17500) * 3.1415926 / 180 / 84.13) * 3.1415926 / 180
        imu_msg.orientation.z = (df['servo'][row] - 17500) * 26 * 3.1415926 / 180 / 6000
         

        bag.write("/CAN_SIGNALS", imu_msg, now)

        '''
        # Odometry: vehicle speed is in m/s, servo angle is in rad

        odometry_msg = Vector3Stamped()
        odometry_msg.header.stamp = now
        
        # x is vehicle speed, y is servo angle
        odometry_msg.vector.x = df['motor'][row] * 0.00024868693
        odometry_msg.vector.y = (df['servo'][row] - 17500) * 0.00008726646

        bag.write("/odometry", odometry_msg, now)
        '''
