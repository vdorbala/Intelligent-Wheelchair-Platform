#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int32MultiArray.h>
#include "math.h"
#include "std_msgs/String.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cmath>
#define radius_of_wheel .15915
#define distance_between_wheels .5 
#define left_counts_per_meter 360
#define right_counts_per_meter 360
//All measurements in meters.
double x = 0.0;
double y = 0.0;
double th =0.0;
double vx = 0.0;                                                      //linear velocity of robot in x direction (m/s)
double vy = 0.0;                                                      //linear velocity of robot in y direction (m/s)
double vth =0.0;                                                      //rate of change of theta (th) in (radians/s)
int i = 0;
float left_counts, right_counts, L_ticks, R_ticks, last_left, last_right ;
float left_distance, right_distance, distance,total_distance;


void call_back(const std_msgs::Int32MultiArray::ConstPtr& speed)
{   
  i = i+1;
    right_counts = -speed->data[0];
    left_counts = speed->data[1]; 

  if ((i%10000) == 0){
   ROS_INFO("r :%d",right_counts);
   ROS_INFO("l :%d",left_counts);
  }
 
 L_ticks=left_counts-last_left;
 R_ticks=right_counts-last_right;
//ROS_INFO("r :%d",R_ticks);
// ROS_INFO("l :%d",L_ticks);
 
    
 left_distance=L_ticks / left_counts_per_meter;
 right_distance=R_ticks / right_counts_per_meter;
//ROS_INFO("r :%f",right_distance);
//  ROS_INFO("l :%f",left_distance);

  distance=(left_distance+right_distance)/2;
  total_distance += distance;

  if ((i%50) == 0 ) {
  ROS_INFO("total distance  :%f meters",total_distance);
}

}


geometry_msgs::PoseStamped constructPoseStampedMsg(float xPos, float yPos, float angle)
{
        geometry_msgs::PoseStamped poseMsg;
        poseMsg.header.frame_id = "/odom";
        poseMsg.header.stamp = ros::Time::now();
        poseMsg.pose.position.x = xPos;
        poseMsg.pose.position.y = yPos;
        poseMsg.pose.position.z = 0;
        tf::Quaternion quat = tf::createQuaternionFromYaw(angle);
        tf::quaternionTFToMsg(quat, poseMsg.pose.orientation);
        return poseMsg;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "odometry");
  
  ros::NodeHandle n;
  ros::NodeHandle o;
  ros::Subscriber odom_sub=o.subscribe("/speed_counter",100,call_back);
  ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("odom", 50);
  ros::Publisher pose_pub = n.advertise<geometry_msgs::PoseStamped>("my_pose", 1000);
  tf::TransformBroadcaster odom_broadcaster;

 

  
  ros::Time current_time, last_time;
  current_time = ros::Time::now();
  last_time = ros::Time::now();

  ros::Rate r(50);
  while(n.ok()){

    ros::spinOnce();               // check for incoming messages
    current_time = ros::Time::now();

   //compute odometry in a typical way given the velocities of the robot
    double dt = (current_time - last_time).toSec();
    //ROS_INFO("dt : %f",dt);	
    
   if ((last_right!=right_counts)||(last_left!=left_counts))
{
    th += (right_distance - left_distance) / distance_between_wheels;
    
    x += distance*cos(th);
    y += distance*sin(th);
    vx =distance*cos(th)/dt;
    vy =distance*sin(th)/dt;
    vth= th/dt;
}   
 last_left=left_counts;
 last_right=right_counts;

    ROS_INFO(" X : %f ",x);  
    ROS_INFO(" Y : %f ",y);

    //since all odometry is 6DOF we'll need a quaternion created from yaw
    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(th);

    //first, we'll publish the transform over tf
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = current_time;
    odom_trans.header.frame_id = "odom";
    odom_trans.child_frame_id = "base_footprint";

    odom_trans.transform.translation.x = x;
    odom_trans.transform.translation.y = y;
    odom_trans.transform.translation.z = 0.0;
    odom_trans.transform.rotation = odom_quat;

    //send the transform
    odom_broadcaster.sendTransform(odom_trans);

    //next, we'll publish the odometry message over ROS
    nav_msgs::Odometry odom;
    odom.header.stamp = current_time;
    odom.header.frame_id = "odom";

    //set the position
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation = odom_quat;

    //set the velocity
    odom.child_frame_id = "base_footprint";
    odom.twist.twist.linear.x = vx;
    odom.twist.twist.linear.y = vy;
    odom.twist.twist.angular.z = vth;

    //publish the message
    odom_pub.publish(odom);
     geometry_msgs::PoseStamped poseStamped = constructPoseStampedMsg(x,y,th);
    pose_pub.publish(poseStamped);
    last_time = current_time;
    r.sleep();
  }
}
