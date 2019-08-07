#!/usr/bin/env python


import rospy
import math
import tf
from tf.msg import tfMessage


    
if __name__ == '__main__':
	
    rospy.init_node('robot_tf_publisher', anonymous=True)                                     # nodename
    rospy.loginfo(" broadcasting the robot transforms ")  
    robot_tf_broadcaster = tf.TransformBroadcaster()
        
    rate =  rospy.Rate(100)
    
    while not rospy.is_shutdown():
				
		#getting the current time	
		current_time = rospy.Time.now()
		#publishing a static transform
		robot_tf_broadcaster.sendTransform((0, 0.425, 0.4), (0, 0, 0, 1), current_time,"kinect2_link", "base_link")
		#rospy.loginfo(" broadcasting the static transforms ")  
		rate.sleep()
		
        				    
