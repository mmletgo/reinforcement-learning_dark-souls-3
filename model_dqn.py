import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import os

# experiences replay buffer size
REPLAY_SIZE = 2000
# memory size 1000
# size of minibatch
small_BATCH_SIZE = 16
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

# these are the hyper Parameters for DQN
# discount factor for target Q to caculate the TD aim value
GAMMA = 0.9
# the start value of epsilon E
INITIAL_EPSILON = 0.5
# the final value of epsilon
FINAL_EPSILON = 0.01

class DQN(nn.Module):
    def __init__(self, observation_width, observation_height, action_space):
        super(DQN, self).__init__()
        # self.state_dim = observation_width * observation_height
        # self.state_w = observation_width
        # self.state_h = observation_height
        # self.action_dim = action_space

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(int((observation_width/4) * (observation_height/4) * 64), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_space)

    def forward(self, x):
        print(x.shape)
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = torch.max_pool2d(x, 2)
        print(x.shape)
        x = torch.relu(self.conv2(x))
        print(x.shape)
        x = torch.max_pool2d(x, 2)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = torch.relu(self.fc1(x))
        print(x.shape)
        x = torch.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x


class CNN(nn.Module):
    def __init__(self, height, width, action_space):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        linear_input_size = convh * convw * 64

        self.fc = nn.Linear(linear_input_size, 512)
        self.output = nn.Linear(512, action_space)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc(x))
        # print(x.shape)
        return self.output(x)

class DQNAgent:
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.replay_buffer = deque()
        self.epsilon = INITIAL_EPSILON
        self.model_file = model_file
        self.log_file = log_file

        self.model = CNN(observation_width, observation_height, action_space)
        self.target_model = CNN(observation_width, observation_height, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        if os.path.exists(self.model_file):
            print("model exists, load model\n")
            self.model.load_state_dict(torch.load(self.model_file))
        else:
            print("model doesn't exist, create new one\n")

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            # state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            state = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()

    def store_data(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

    def train_network(self, BATCH_SIZE, num_step):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = torch.FloatTensor([data[0] for data in minibatch]).squeeze(1)
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        next_state_batch = torch.FloatTensor([data[3] for data in minibatch]).squeeze(1)

        q_values = self.model(state_batch)
        next_q_values = self.target_model(next_state_batch)
        q_value = q_values.gather(1, action_batch.argmax(dim=1, keepdim=True)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward_batch + GAMMA * next_q_value * (1 - torch.FloatTensor([data[4] for data in minibatch]))

        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if num_step % 100 == 0:
            print("loss: ", loss.item())

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)
        print("Save to path:", self.model_file)