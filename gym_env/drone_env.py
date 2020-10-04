#!/usr/bin/env python3

# ___________________________________________________ Libraries ___________________________________________________ #


from math import sqrt

import gym
from geometry_msgs.msg import Pose
from gym import spaces, utils
from gym.envs.registration import register

from drone_controller import Drone_Controller


# ___________________________________________ Constants and Definitions ___________________________________________ #

# MAX Lenght of a single episode
episodeBound = 500

# HARD CODING - PIPE AT PREDEFINED LOCATION
PIPE16_LOC = [1.73, -1.46, 1.86]


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
        self.max_x_distance = 1.0
        self.max_y_distance = 0.5
        self.max_z_distance = 0.2
        self.episodeSteps = 0
        self.reset_number = 0
        self.vel_value = 0.1

        self.controller_object = Drone_Controller()

        # Action definition
        #   1. vel_y +
        #   2. vel_y -
        #   3. vel_y /
        self.action_space = spaces.Discrete(3)

        # Observation definition
        high = np.array([
                         np.finfo(np.float).max,    # x_drone_pose - x_des_pose
                         np.finfo(np.float).max,    # y_drone_pose - y_des_pose
                         np.finfo(np.float).max     # z_drone_pose - z_des_pose
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

        # Arm the drone if disarmed
        self.controller_object.armDrone()

        # If this is the first reset, the drone must reach the pipe on point [1.63, -1.46, 3.0]. Otherwise, it must reposition on a point
        # convenient to repeat learning 
        self.reset_number += 1
        if self.reset_number == 1:
            self.controller_object.reachPipe()
        else:
            self.controller_object.reposition()
        
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
        elif action == 2:   # vel_y /
            #print("\nvel_y /")
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
        #print("[", x_drone_distance, "|", y_drone_distance, "|", z_drone_distance, "]")
        norm_distance = sqrt(
                             # at first it doesn't consider x_pose
                             #pow(x_drone_distance, 2) +
                             pow(y_drone_distance, 2) +
                             pow(z_drone_distance, 2)
                            )

        
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
            # NEED TO BE IMPROVED
            reward = 1/(pow(norm_distance, 2))
            """x_pos = self.controller_object.getPosition().x
            y_pos = self.controller_object.getPosition().y
            z_pos = self.controller_object.getPosition().z
            print("[x|y|z] = [", x_pos, "|", y_pos, "|", z_pos, "] -> ", reward)"""
        elif z_drone_distance < 0 and abs(y_drone_distance) <= 0.05:
            print("VERY GOOD!")
            reward = 10000
        else: 
            reward = 0
        
        return observation, reward, done, {}
