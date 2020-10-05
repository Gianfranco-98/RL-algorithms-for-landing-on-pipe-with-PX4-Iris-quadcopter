#!/usr/bin/env python3

# ___________________________________________________ Libraries ___________________________________________________ #


import time
from collections import namedtuple
import math
from math import sqrt
from decimal import *

import gym
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from gym import spaces, utils
from gym.envs.registration import register

from drone_controller import Drone_Controller


# ___________________________________________ Constants and Definitions ___________________________________________ #

# MAX Lenght of a single episode
episodeBound = 500

# HARD CODING - PIPE AT PREDEFINED LOCATION
PIPE16_LOC = [1.73, -1.46, 1.86]

# HYPERPARAMETERS
MAX_X_DIST = 1.0
MAX_Y_DIST = 0.5
MAX_Z_DIST = 0.2
VELOCITY_VALUE = 0.1


# _____________________________________________ Classes and Functions _____________________________________________ #


# Register our custom industrial environment
reg = register(
    id='IndustrialDrone-v0',
    entry_point='drone_env:DroneEnv',
    max_episode_steps=episodeBound,
    )


class DroneEnv(gym.Env):

    def __init__(self):

        # Setting env parameters [can be achieved also with parameter server]
        self.desired_pose = Pose()
        self.desired_pose.position.x = PIPE16_LOC[0]
        self.desired_pose.position.y = PIPE16_LOC[1]
        self.desired_pose.position.z = PIPE16_LOC[2]    
        self.max_x_distance = MAX_X_DIST
        self.max_y_distance = MAX_Y_DIST
        self.max_z_distance = MAX_Z_DIST
        self.vel_value = VELOCITY_VALUE
        self.episodeSteps = 0

        self.controller_object = Drone_Controller()

        # Action definition
        #   1. vel_y +
        #   2. vel_y -
        #   3. vel_y 0
        self.action_space = spaces.Discrete(3)

        # Observation definition
        high = np.array([
                         self.max_x_distance,    # x_drone_pose - x_des_pose
                         self.max_y_distance,    # y_drone_pose - y_des_pose
                         self.max_z_distance     # z_drone_pose - z_des_pose
                       ])
        self.observation_space = spaces.Box(-high, high)

        print("Initializing drone...")
        print("Arming the drone...")
        self.controller_object.armDrone()               
        print("Activating Offboard...")
        self.controller_object.activateOffboard()


    def reset(self):

        # Reset episode steps
        self.episodeSteps = 0

        # Reach episode starting point
        self.controller_object.reachPipe()
        
        # Get observation 
        x_dist = (
                  self.controller_object.getPosition().x - 
                  self.desired_pose.position.x
                 )
        y_dist = (
                  self.controller_object.getPosition().y - 
                  self.desired_pose.position.y
                 )
        z_dist = (
                  self.controller_object.getPosition().z - 
                  (self.desired_pose.position.z + self.max_z_distance)
                 )
        observation = [
                       x_dist, 
                       y_dist,
                       z_dist
                      ]

        # Initialize velocity
        self.controller_object.initVelocity()

        return observation


    def step(self, action):


        # Action select

        if action == 0:     # vel_y +
            #print("\nvel_y +")
            self.controller_object.controlVelocity(
                                                   0,
                                                   self.vel_value
                                                  )
        elif action == 1:   # vel_y -
            #print("\nvel_y -")
            self.controller_object.controlVelocity(
                                                   0,
                                                   -self.vel_value
                                                  ) 
        elif action == 2:   # vel_y 0
            #print("\nvel_y 0")
            self.controller_object.controlVelocity(
                                                   0,
                                                   0
                                                  )


        # Get new state

        x_dist = (
                  self.controller_object.getPosition().x - 
                  self.desired_pose.position.x
                 )
        y_dist = (
                  self.controller_object.getPosition().y - 
                  self.desired_pose.position.y
                 )
        z_dist = (
                  self.controller_object.getPosition().z - 
                  (self.desired_pose.position.z + self.max_z_distance)
                 )
        observation = [
                       x_dist, 
                       y_dist,
                       z_dist
                      ]                            
        x_drone_distance = observation[0]
        y_drone_distance = observation[1]
        z_drone_distance = observation[2]
        
        # Calculating reward and done

        done = False
        reward = 0.0
        self.episodeSteps += 1
    
        if (
            abs(x_drone_distance) > self.max_x_distance or 
            abs(y_drone_distance) > self.max_y_distance or 
            z_drone_distance < 0 or 
            self.episodeSteps > episodeBound
           ):
           done = True

        if not done:
            reward = 1/(abs(y_drone_distance) + 0.001) # 0.001 to not diverge
        elif z_drone_distance < 0 and abs(y_drone_distance) < self.max_y_distance: 
            reward = 0
        else:
            reward = -100
        return observation, reward, done, {}