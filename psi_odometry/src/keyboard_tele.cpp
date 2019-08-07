#include "ros/ros.h"
#include "std_msgs/String.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <geometry_msgs/Twist.h>
#include "psi_odometry/RoboteqDevice.h"
#include "psi_odometry/ErrorCodes.h"
#include "psi_odometry/Constants.h"
#include "std_msgs/Int32MultiArray.h"
//variables
//w1 --> speed in rpm of motor1 i.e right motor
//w2 --> speed in rpm of motor2 i.e left motor
//c1 --> no. of counts of motor1
//c2 --> no. of counts of motor2
using namespace std;
RoboteqDevice device;
float lin_vel=0;
float ang_vel=0;
int w1,w2=0;
int c1,c2=0;

/*void motor()                                               //function for separate mode
{    
 int status=0;
 int m1,m2=0;

 if (ang_vel==2)
  {	ROS_INFO("in loop a1");
    m1=300;
    m2=-300;
  }
 
 else if (ang_vel==-2)
  {	ROS_INFO("in loop a2");
    m1=-300;
    m2=300;
  }

     
 if (lin_vel==2)
  {	ROS_INFO("in loop l1");
    m1=300;
    m2=300;
  }
  
 else if (lin_vel==-2)
 {	ROS_INFO("in loop l2");
   m1=-300;
   m2=-300;
 }
 

 m1=-m1;
 m2=m2;
 if((status = device.SetCommand(_G, 1, m1) != RQ_SUCCESS))
        {
		ROS_INFO("Failed... Error code --> ", status);
	}
 if((status = device.SetCommand(_G, 2, m2) != RQ_SUCCESS))
       {		ROS_INFO("Failed... Error code --> ", status);
	}


}*/

void motor()                                       // function for mixed mode
{    
 int status=0;
 int m1,m2=0;
 

 if (ang_vel==2)
  {	ROS_INFO("in loop a1");
    m2=0;
    m1=100;
  }
 
 else if (ang_vel==-2)
  {	ROS_INFO("in loop a2");
    m2=0;
    m1=-100;
  }

     
 if (lin_vel==-2)
  {	ROS_INFO("in loop l1");
    m2=150;
    m1=0;
  }
  
 else if (lin_vel==2)
 {	ROS_INFO("in loop l2");
   m2=-150;
   m1=0;
 }

 m1=-m1;
 m2=m2;
 if((status = device.SetCommand(_G, 1, m1) != RQ_SUCCESS))                   //command to run motor1
        {
		ROS_INFO("Failed... Error code --> ", status);
	}
 if((status = device.SetCommand(_G, 2, m2) != RQ_SUCCESS))                   //command to run motor2 
       {		ROS_INFO("Failed... Error code --> ", status);
	}

/*if((status = device.GetValue(_S, 1, w1) !=RQ_SUCCESS))                     //command to get speed(rpm) of motor1  
	{
		ROS_INFO("Failed..... Error Code---->", status);
	}
		

	if((status = device.GetValue(_S, 2, w2) !=RQ_SUCCESS))              //command to get speed(rpm) of motor2
	{
		ROS_INFO("Failed..... Error Code---->", status);
	}*/
		

if((status = device.GetValue( _C, 1 , c1) != RQ_SUCCESS)) {                  //command to get counts of motor1
		ROS_INFO("Failed... Error code --> ", status);
	}
  if((status = device.GetValue( _C, 2 , c2) != RQ_SUCCESS)) {                //command to get counts of motor2
		ROS_INFO("Failed... Error code --> ", status);
	}



}



void cmd_velocityCallback (const geometry_msgs::Twist::ConstPtr& twist)
{ 
  lin_vel = twist->linear.x;
  ang_vel = twist->angular.z;
  cout<<"linear velocity= "<<lin_vel<<endl;
  cout<<"angular velocity= "<<ang_vel<<endl;
  motor();
  
}



   
int main(int argc, char *argv[])
{
    
        ros::init(argc, argv, "keyboard_tele");
   
        string response = "";
	ROS_INFO("Initializing...");
	usleep(500000);
	
	int status = device.Connect("/dev/ttyACM0");

	while (status != RQ_SUCCESS && ros::ok())
	{
		ROS_INFO("Error connecting to device: ", status, "\n");
		ROS_INFO("Attempting server restart...");
		usleep(999999);
		device.Disconnect();

		status = device.Connect("/dev/ttyACM0");
		if (status == 0) 
          { 
			ROS_INFO("Connection re-established...");
          }
	}

	if((status = device.SetConfig(_MXMD, 1)) != RQ_SUCCESS)          // set in mixed mode
		cout<<"failed --> "<<status<<endl;
	else
		cout<<"succeeded."<<endl;



        if((status = device.SetConfig(_MMOD, 1, 1)) != RQ_SUCCESS)          // set in closed loop channel 1
		cout<<"failed --> "<<status<<endl;
	else
		cout<<"succeeded."<<endl;

        if((status = device.SetConfig(_MMOD, 2, 1)) != RQ_SUCCESS)          // set in closed loop channel 2
		cout<<"failed --> "<<status<<endl;
	else
		cout<<"succeeded."<<endl;

       if(status= device.SetCommand(_SENCNTR,1,0) !=RQ_SUCCESS)       // set encoder of motor1 to zero 
                 cout<<"failed --> "<<status<<endl;
        else
		cout<<"succeeded."<<endl;
 
    if(status= device.SetCommand(_SENCNTR,2,0) !=RQ_SUCCESS)       // set encoder of motor2 to zero 
                 cout<<"failed --> "<<status<<endl;
        else
		cout<<"succeeded."<<endl;

	//ros::NodeHandle nh_;
	ros::NodeHandle o;
        ros::NodeHandle n;
	//ros::Subscriber sub = nh_.subscribe("/turtle1/cmd_vel", 1, cmd_velocityCallback);
	ros::Subscriber sub = n.subscribe("/cmd_vel",1,cmd_velocityCallback);
	ros::Publisher info = o.advertise<std_msgs::Int32MultiArray>("speed_counter",1);
	while(ros::ok())
{
	std_msgs::Int32MultiArray speed;
	
	speed.data.clear();
	
	
	speed.data.push_back(c1);
	speed.data.push_back(c2);
	info.publish(speed);
      
	ros::spinOnce();
	
}	
	return 0;
}
