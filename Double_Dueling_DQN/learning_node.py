#!/usr/bin/env python3


# ___________________________________________ Libraries and Definitions ___________________________________________ #


# ROS management
import rospy

# Useful for dictionaries and tuples
import collections

# Math functions
import numpy as np
import random

# Learning libraries
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import dqn_model

# Custom environment
import drone_env

# Visual results in tensorboard
from tensorboardX import SummaryWriter

# ___________________________________________________ Constants ___________________________________________________ #


# Learning parameters
GAMMA = 0.99
LEARNING_RATE = 0.01
EPSILON_START = 1.0
EPSILON_DECAY_STEPS = 1000
EPSILON_FINAL = 0.05
SYNC_NETS_STEPS = 20
BATCH_SIZE = 128
MEAN_REWARD_BOUND = 5000
DOUBLE_DQN_ACTIVE = True

# Buffer parameters
EXP_REPLAY_BUFFER_SIZE = 200
SAMPLE_START_STEPS = 200

# Gym Environment parameters
ENV_NAME = "IndustrialDrone-v0"

# Experience definition
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


# _____________________________________________ Classes ad Functions _____________________________________________ #


class Exp_Replay_Buffer:

    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, sample_size):
        buffer_lenght = len(self.buffer)
        sample_places = np.random.choice(len(self.buffer), sample_size,
                                   replace=False)

        # Take samples
        states = []
        actions = [] 
        rewards = []
        dones = []
        new_states = []
        for index in sample_places:
            states.append(self.buffer[index].state)
            actions.append(self.buffer[index].action)
            rewards.append(self.buffer[index].reward)
            dones.append(self.buffer[index].done)
            new_states.append(self.buffer[index].new_state)

        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(new_states)

class DQN_Agent:

    def __init__(self, env, exp_replay_buffer):
        self.env = env
        self.exp_replay_buffer = exp_replay_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        final_reward = None
        state_array = np.array([self.state], copy=False)
        state_tensor = torch.tensor(state_array).to(device)
        q_values_tensor = net(state_tensor.float())
        _, action_tensor = torch.max(q_values_tensor, dim=1)
        best_action = int(action_tensor.item())
        random_action = np.random.choice(self.env.action_space.n)
        action = np.random.choice([best_action, random_action], p=[1-epsilon, epsilon])

        # Step
        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward,
                         done, new_state)
        self.exp_replay_buffer.append(exp)
        self.state = new_state
        if done:
            final_reward = self.total_reward
            self._reset()

        return final_reward

def calc_loss(batch, net, tgt_net, device="cpu", double=True):

    # Collecting experiences from batch
    states, actions, rewards, dones, new_states = batch
    states_tensor = torch.tensor(np.array(
        states, copy=False)).to(device)
    actions_tensor = torch.tensor(actions).to(device)
    rewards_tensor = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    state_action_values = net(states_tensor.float()).gather(
        1, actions_tensor.unsqueeze(-1)).squeeze(-1)

    # Bellman approximation with Double Q-Learning
    with torch.no_grad():
        new_states_tensor = (torch.tensor(np.array(
                         new_states, copy=False)).float()).to(device)
        if double:
            new_state_actions = net(new_states_tensor.float()).max(1)[1]
            new_state_values = tgt_net(new_states_tensor.float()).gather(
                1, new_state_actions.unsqueeze(-1)).squeeze(-1)
        else:
            new_state_values = tgt_net(new_states_tensor.float()).max(1)[0]
        new_state_values[done_mask] = 0.0
        new_state_values = new_state_values.detach()

        expected_state_action_values = new_state_values * GAMMA + \
                                   rewards_tensor

    # Applying loss
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


# _____________________________________________________ Main _____________________________________________________ #


# Main function
def main():

    # Initialize the node
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

    # Tensorboard writer
    writer = SummaryWriter(comment = "__" + ENV_NAME + "__")

    # Agent initialization
    replay_buffer = Exp_Replay_Buffer(EXP_REPLAY_BUFFER_SIZE)
    agent = DQN_Agent(env, replay_buffer)

    # Other initialization
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON_START
    total_rewards = []
    iterations = 0
    best_mean_reward = None

    # ROS main loop
    while not rospy.is_shutdown():

        # Epsilon decay
        iterations += 1
        if epsilon <= EPSILON_FINAL:
            epsilon = EPSILON_FINAL
        else:
            epsilon = EPSILON_START - iterations/EPSILON_DECAY_STEPS

        # Step
        reward = agent.play_step(net, epsilon, device=device)

        # Store informations
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print("[%d] - done %d games, reward_mean %.3f, "
                  "eps %.2f" % (
                iterations, len(total_rewards), mean_reward, epsilon
            ))
            writer.add_scalar("epsilon", epsilon, iterations)
            writer.add_scalar("reward_100", mean_reward, iterations)
            writer.add_scalar("reward", reward, iterations)
            if best_mean_reward is None or best_mean_reward < mean_reward:
            #    Uncommenting, we save net weights and best rewards in a PATH's file
            #    torch.save(net.state_dict(), PATH)
                if best_mean_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d iterations!" % iterations)
                break
                
        # Fill a little the buffer before start optimization
        if len(replay_buffer) < SAMPLE_START_STEPS:
            continue

        # Periodic synchronization of nets
        if iterations % SYNC_NETS_STEPS == 0:
            tgt_net.load_state_dict(net.state_dict())

        # Optimization
        optimizer.zero_grad()
        batch = replay_buffer.sample(BATCH_SIZE)
        error = calc_loss(batch, net, tgt_net, device=device, double=DOUBLE_DQN_ACTIVE)
        writer.add_scalar("loss", error, iterations)
        error.backward()
        optimizer.step()

    writer.close()


if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass