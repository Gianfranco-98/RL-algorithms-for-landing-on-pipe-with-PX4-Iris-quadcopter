#!/usr/bin/env python3


# ___________________________________________________ Libraries ___________________________________________________ #


# Useful to manage ROS with Python language
import rospy

# MAVROS msgs to use setpoints
from geometry_msgs.msg import Point, PoseStamped, Twist, Vector3
from sensor_msgs.msg import Imu
from mavros_msgs.msg import *

# MAVROS srv to change modes 
from mavros_msgs.srv import *

# Math functions
from math import sin,cos,sqrt
import numpy as np


# ___________________________________________________ Constants ___________________________________________________ #


# HARD CODING - PIPE AT PREDEFINED LOCATION
PIPE16_LOC = [1.73, -1.46, 1.86]

# Ideal repositioning altitude
ALTITUDE = 3.0

# Thrust to go up slowly
THRUST = 0.60


# ____________________________________________________ Classes ____________________________________________________ #


# Flight modes class
class Drone_Modes:

    def __init__(self):
        pass

    def setArm(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(True)
        except rospy.ServiceException as e:
            print("Service arm call failed: %s"%e)

    def setOffboardMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='OFFBOARD')
        except rospy.ServiceException as e:
            print("service set_mode call failed: %s. Offboard Mode could not be set."%e)


# Controller class
class Drone_Controller:

    def __init__(self):

        #           Setpoint message for position control
        # ---------------------------------------------------------
        ## Message
        self.sp_pos = PositionTarget()
        ## Bitmask to use only Position control
        self.sp_pos.type_mask = int('111111111000', 2)
        ## Coordinate system: LOCAL_NED
        self.sp_pos.coordinate_frame = 1
        ## Initial position values
        self.sp_pos.position.x = 0.0
        self.sp_pos.position.y = 0.0
        self.sp_pos.position.z = 0.0
        # Message for the actual local position of the drone
        self.local_pos = Point(0.0, 0.0, 0.0)
        # ---------------------------------------------------------

        #           Setpoint message for velocity control
        # ---------------------------------------------------------
        ## Message
        self.sp_vel = Twist()
        ## Initial velocity values
        self.sp_vel.angular.x = 0
        self.sp_vel.angular.y = 0
        self.sp_vel.angular.z = 0
        self.sp_vel.linear.x = 0
        self.sp_vel.linear.y = 0
        self.sp_vel.linear.z = 0
        # Message for the actual local velocity of the drone
        self.velocity = Point(0.0, 0.0, 0.0)
        # ---------------------------------------------------------

        #           Setpoint message for velocity control
        # ---------------------------------------------------------
        # Message
        self.sp_att = AttitudeTarget()
        # Bitmask to use all
        self.sp_att.type_mask = int('00000000', 2)
        # Rate values [rads/second]
        self.sp_att.body_rate.x = 0.09
        self.sp_att.body_rate.y = 0.09
        self.sp_att.body_rate.z = 0.09
        # Initial thrust
        self.sp_att.thrust = 0.0
        # Initial orientation values [quaternion form]
        self.sp_att.orientation.x = 0.0
        self.sp_att.orientation.y = 0.0
        self.sp_att.orientation.z = 0.0
        self.sp_att.orientation.w = 0.0

        # ---------------------------------------------------------


        # Drone state
        self.state = State()

        # Drone modes
        self.modes = Drone_Modes()

        # Updating rate
        self.rate = rospy.Rate(20)

        # Setpoint_raw publisher for position control   
        self.sp_raw_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)

        # Setpoint_velocity publisher for velocity control
        self.sp_vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)

        # Setpoint_raw attitude publisher
        self.sp_att_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)

        # Subscribe to drone state
        rospy.Subscriber('mavros/state', State, self.stateCb)   

        # Subscribe to drone's local position
        rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.posCb)

        # Subscribe to drone's local velocity
        rospy.Subscriber('mavros/setpoint_raw/target_local', PositionTarget, self.velCb)


    # ____________ Utility functions____________

    ## Get actual position
    def getPosition(self):
        return self.local_pos

    ## Get actual velocity
    def getVelocity(self):
        return self.velocity

    ## Initialize velocity
    def initVelocity(self):
        self.sp_vel.linear.x = 0.0
        self.sp_vel.linear.y = 0.0
        self.sp_vel.linear.z = -0.1

    ## Publish target velocity
    def pubVelocity(self):
        self.sp_vel_pub.publish(self.sp_vel)
        self.rate.sleep()

    ## Publish target position
    def pubPosition(self):
        self.sp_raw_pub.publish(self.sp_pos)
        self.rate.sleep()

    ## Publish target attitude
    def pubAttitude(self):
        self.sp_att_pub.publish(self.sp_att)
        self.rate.sleep()

    ## Arm the drone if not armed
    def armDrone(self):
        if not self.state.armed:
            while not self.state.armed:
                self.modes.setArm()
                self.rate.sleep()

    ## Activate OFFBOARD Mode by sending a few setpoints
    def activateOffboard(self):
        for _ in range(10):
            self.sp_raw_pub.publish(self.sp_pos)
            self.rate.sleep()
        self.modes.setOffboardMode()

    ## Return true if x pose has been reached (+-2 tolerance)
    def x_reached(self):
        if self.local_pos.x > PIPE16_LOC[0]-2 and self.local_pos.x < PIPE16_LOC[0]+2:
            return True
        else:
            return False

    ## Return true if y pose has been reached (+-0.02 tolerance)
    def y_reached(self):
        if self.local_pos.y > PIPE16_LOC[1]-0.02 and self.local_pos.y < PIPE16_LOC[1]+0.02:
            return True
        else:
            return False

    ## Return true if x and y poses have been reached
    def pipe_reached(self):
        if self.x_reached() == True and self.y_reached() == True:
            return True
        else:
            return False  

    ## Return true if drone is repositioned to learn
    def repositioned(self):
        if (
            self.local_pos.x > PIPE16_LOC[0]-1 and self.local_pos.x < PIPE16_LOC[0]+1 and
            self.local_pos.y > PIPE16_LOC[1]-0.5 and self.local_pos.y < PIPE16_LOC[1]+0.5 and
            self.local_pos.z > PIPE16_LOC[2]+1 and self.local_pos.z < ALTITUDE + 0.5
            ):
            return True
        else:
            return False

    ## Attitude control function
    def controlAttitude(self, thrust, angle, versor):
        norm_versor = sqrt(versor[0]*versor[0] + versor[1]*versor[1] + versor[2]*versor[2])
        self.sp_att.header.seq += 1
        self.sp_att.header.stamp = rospy.Time.now()
        self.sp_att.header.frame_id = "controlAttitude"
        self.sp_att.thrust = thrust
        self.sp_att.orientation.x = sin(angle/2)*versor[0]/norm_versor
        self.sp_att.orientation.y = sin(angle/2)*versor[1]/norm_versor
        self.sp_att.orientation.z = sin(angle/2)*versor[2]/norm_versor
        self.sp_att.orientation.w = cos(angle/2)
        self.pubAttitude()

    ## Velocity control function
    def controlVelocity(self, x_value, y_value):
        self.sp_vel.linear.x = x_value
        self.sp_vel.linear.y = y_value
        self.pubVelocity()

    ## Reach the pipe and maintain position
    def controlPosition(self):
        self.sp_pos.header.seq += 1
        self.sp_pos.header.stamp = rospy.Time.now()
        self.sp_pos.header.frame_id = "controlPosition"
        self.sp_pos.position.x = PIPE16_LOC[0]
        self.sp_pos.position.y = PIPE16_LOC[1]
        self.sp_pos.position.z = ALTITUDE
        self.pubPosition()

    ## Reach the pipe from start point
    def reachPipe(self):
        while not self.pipe_reached():
            self.controlPosition()

    ## Reposition on pipe
    def reposition(self):
        while not self.repositioned():
            self.controlPosition()
            versor = [0.0, 0.0, 1.0]
            angle = 0
            self.controlAttitude(THRUST, angle, versor)

    # ___________________________________________ 



    # ____________ Callback functions ____________ 

    ## local position callback
    def posCb(self, msg):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z 

    ## velocity callback
    def velCb(self, msg):
        self.velocity.x = msg.velocity.x
        self.velocity.y = msg.velocity.y
        self.velocity.z = msg.velocity.z

    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg

    # ___________________________________________ 
