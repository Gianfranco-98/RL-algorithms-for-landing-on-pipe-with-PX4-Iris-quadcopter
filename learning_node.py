#!/usr/bin/env python3


# ___________________________________________ Libraries and Definitions ___________________________________________ #


# ROS management
import rospy
import rospkg

# Useful for dictionaries and tuples
import collections

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
import dqn_model

# Time management
import time

# Custom environment
import drone_env

# Visual results in tensorboard
from tensorboardX import SummaryWriter

# ___________________________________________________ Constants ___________________________________________________ #


# HARD CODING - PIPE AT PREDEFINED LOCATION
PIPE16_LOC = [1.73, -1.46, 1.86]

# Learning parameters
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = 1e-4
SYNC_TARGET_ITERS = 1000
REPLAY_BUFFER_START_SIZE = 10000
EPSILON_DECAY_LAST_ITER = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
MEAN_REWARD_BOUND = 50000

# Gym Environment parameters
ENV_NAME = "IndustrialDrone-v0"

# Experience definition
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


# _____________________________________________ Classes ad Functions _____________________________________________ #


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):

        # Obtaining batch_size samples from buffer
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])

        # Returning all informations about captured experiences
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    # play step, but don't calculate gradients (to save memory)
    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        
        done_reward = None

        # Select action
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v.float())
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # Do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # Update replay buffer
        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):

    # Information gathering from batch
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = (torch.tensor(np.array(
        next_states, copy=False)).float()).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    state_action_values = net(states_v.float()).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)

    # Obtaining q_vals
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v.float()).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    # Bellman approssimation
    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_v

    # Applying loss
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


# _____________________________________________________ Main _____________________________________________________ #


# Main function
def main():

    # Initialize the ROS node
    rospy.init_node('Learning_Node', anonymous=True)
    device = torch.device("cpu")    # Alternatively can be used "cuda" 
    env = gym.make(ENV_NAME)

    # We use two networks synchronised each  to avoid to prevent updating 
    # the weights of one state from too much influence the weights of neighboring states, 
    # making learning unstable
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)
    net = net.float()
    tgt_net = tgt_net.float()

    writer = SummaryWriter(comment="-" + ENV_NAME)

    buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    iterations = 0
    best_m_reward = None

    # ROS main loop
    while not rospy.is_shutdown():

        iterations += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      iterations / EPSILON_DECAY_LAST_ITER)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward_mean %.3f, "
                  "eps %.2f" % (
                iterations, len(total_rewards), m_reward, epsilon
            ))
            writer.add_scalar("epsilon", epsilon, iterations)
            writer.add_scalar("reward_100", m_reward, iterations)
            writer.add_scalar("reward", reward, iterations)
            if best_m_reward is None or best_m_reward < m_reward:
            #    Uncommenting, we save net weights and best rewards
            #    torch.save(net.state_dict(), args.env +
            #               "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d iterations!" % iterations)
                break
                
        if len(buffer) < REPLAY_BUFFER_START_SIZE:
            continue

        if iterations % SYNC_TARGET_ITERS == 0:
            tgt_net.load_state_dict(net.state_dict())

        # Optimization
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        writer.add_scalar("loss", loss_t, iterations)
        loss_t.backward()
        optimizer.step()
    writer.close()


if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
