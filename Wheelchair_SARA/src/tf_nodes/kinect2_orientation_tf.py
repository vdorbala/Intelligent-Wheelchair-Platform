#!/usr/bin/env python


import rospy
import math
import tf
from tf.msg import tfMessage


    
if __name__ == '__main__':
	
    rospy.init_node('kinect2_orientation_tf', anonymous=True)                                     # nodename
    rospy.loginfo(" broadcasting the kinect transforms ")  
    br = tf.TransformBroadcaster()
        
    rate =  rospy.Rate(100)
    
    while not rospy.is_shutdown():
				
		#getting the current time	
		current_time = rospy.Time.now()
		#publishing a static transform
		br.sendTransform((0, 0, 0), (-0.707, 0, 0, 0.707), current_time,"kinect2_link", "sensor_link")
		#rospy.loginfo(" broadcasting the kinect transforms ")  
		rate.sleep()
		
        				    
