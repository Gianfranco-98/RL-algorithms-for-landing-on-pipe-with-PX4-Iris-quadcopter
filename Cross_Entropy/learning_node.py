#!/usr/bin/env python3


# ___________________________________________ Libraries and Definitions ___________________________________________ #


# ROS management
import rospy
import rospkg

# Useful for dictionaries and tuples
from collections import namedtuple

# MAVROS msgs to use setpoints
from geometry_msgs.msg import Point, PoseStamped
from mavros_msgs.msg import *

# MAVROS srv to change modes 
from mavros_msgs.srv import *

# Math functions
import numpy as np
from math import sin,cos,sqrt
import random

# Learning libraries
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# Custom environment
import drone_env

# Episode definition
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])


# ___________________________________________________ Constants ___________________________________________________ #


# Learning Hyperparameters
HIDDEN_SIZE = 64
BATCH_SIZE = 2
PERCENTILE = 70
LEARNING_RATE = 0.01

# Epsilon-Greedy parameters
EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY_STEPS = 10000

# Gym Environment parameters
ENV_NAME = "IndustrialDrone-v0"

# Global variable to add steps in TB writers
steps = 0


# _____________________________________________ Classes ad Functions _____________________________________________ #


# Simple Neural Network
class Net(nn.Module):

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
        
# Cross Entropy Agent
class CE_Agent:

    def __init__(self, env):
        self.env = env
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.episode_steps = []
    
    def batch_playing(self, epsilon, net, batch_size):
        global steps
        batch = []
        self._reset()
        state = self.state
        sm = nn.Softmax(dim=1)

        # Iterate batch episodes
        while True:
            state_tensor = torch.FloatTensor([state])
            action_probs_tensor = sm(net(state_tensor))
            action_probs = action_probs_tensor.data.numpy()[0]
            best_action = np.random.choice(len(action_probs), p=action_probs)
            rand_action = np.random.choice(len(action_probs))
            action = np.random.choice([best_action, rand_action], p=[1-epsilon, epsilon])
            new_state, reward, done, _ = self.env.step(action)
            steps += 1
            self.state = new_state
            self.episode_reward += reward
            step = EpisodeStep(state=state, action=action)
            self.episode_steps.append(step)
            if done:
                print("\n________Total reward = ", self.episode_reward, "________\n")
                episode = Episode(reward=self.episode_reward, steps=self.episode_steps)
                batch.append(episode)
                self._reset()
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            state = self.state

def batch_filtering(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    train_states = []
    train_actions = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_states.extend(map(lambda step: step.state, steps))
        train_actions.extend(map(lambda step: step.action, steps))
    train_states_tensor = torch.FloatTensor(train_states)
    train_actions_tensor = torch.LongTensor(train_actions)
    return train_states_tensor, train_actions_tensor, reward_bound, reward_mean


# _____________________________________________________ Main _____________________________________________________ #


# Main function
def main():

    # Initialize node and env
    rospy.init_node('Learning_Node', anonymous=True)
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize learning objects
    net = Net(state_size, HIDDEN_SIZE, n_actions)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON_START
    agent = CE_Agent(env)

    # Tensorboard Writer
    writer = SummaryWriter(comment = "__" + ENV_NAME + "__")

    # ROS main loop
    while not rospy.is_shutdown():

        # Epsilon decay
        if epsilon <= EPSILON_FINAL:
            epsilon = EPSILON_FINAL
        else:
            epsilon = EPSILON_START - steps/EPSILON_DECAY_STEPS

        # Training loop
        for iter_no, batch in enumerate(agent.batch_playing(
            epsilon, net, BATCH_SIZE)):
            states_tensor, actions_tensor, reward_bound, reward_mean = \
                batch_filtering(batch, PERCENTILE)
            optimizer.zero_grad()
            action_scores_tensor = net(states_tensor)
            error = loss_function(action_scores_tensor, actions_tensor)
            error.backward()
            optimizer.step()
            print("%d [step %d]: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
                iter_no, steps, error.item(), reward_mean, reward_bound))
            writer.add_scalar("loss", error.item(), steps)
            writer.add_scalar("reward_bound", reward_bound, steps)
            writer.add_scalar("reward_mean", reward_mean, steps)
            if reward_mean > 5000:
                print("Solved in %f Batches [%d steps]!" % (iter_no, steps))
                break

    writer.close()

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
