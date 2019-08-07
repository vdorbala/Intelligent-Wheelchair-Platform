#!/usr/bin/env python


import rospy
import math
import tf
from tf.msg import tfMessage


    
if __name__ == '__main__':
	
    rospy.init_node('laser_link', anonymous=True)                                     # nodename
    rospy.loginfo(" broadcasting the laser transforms ")  
    laser_br = tf.TransformBroadcaster()
        
    rate =  rospy.Rate(100)
    
    while not rospy.is_shutdown():
				
		#getting the current time	
		current_time = rospy.Time.now()
		#publishing a static transform
                #  (0.5, -0.5, 0.5, 0.5) first y-> -90 degrees then x -> 90 degrees  
		laser_br.sendTransform((0, 0, 0),(0.5, -0.5, 0.5, 0.5), current_time,"laser_link", "kinect2_link")
		#rospy.loginfo(" broadcasting the kinect transforms ")  
		rate.sleep()
		
        				    
