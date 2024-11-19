import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import os
from typing import Tuple

class Config:
    REPLAY_SIZE = 2000
    SMALL_BATCH_SIZE = 16
    BIG_BATCH_SIZE = 128
    BATCH_SIZE_DOOR = 1000

    GAMMA = 0.9
    INITIAL_EPSILON = 0.5  
    FINAL_EPSILON = 0.01  
    EPSILON_DECAY_STEPS = 10000  
    
    LEARNING_RATE = 1e-3
    TARGET_UPDATE_FREQUENCY = 100  
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, height: int, width: int, action_space: int):
        super(CNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积层输出大小
        def conv2d_size_out(size: int, kernel_size: int, stride: int) -> int:
            return (size - (kernel_size - 1) - 1) // stride + 1

        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        linear_input_size = convh * convw * 64

        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.value_stream(features)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(states).squeeze(1).to(Config.DEVICE),
                torch.FloatTensor(actions).to(Config.DEVICE),
                torch.FloatTensor(rewards).to(Config.DEVICE),
                torch.FloatTensor(next_states).squeeze(1).to(Config.DEVICE),
                torch.FloatTensor(dones).to(Config.DEVICE))

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(self, observation_width: int, observation_height: int, 
                 action_space: int, model_file: str, log_file: str):
        self.action_dim = action_space
        self.epsilon = Config.INITIAL_EPSILON
        self.model_file = model_file
        self.log_file = log_file

        self.model = CNN(observation_width, observation_height, action_space).to(Config.DEVICE)
        self.target_model = CNN(observation_width, observation_height, action_space).to(Config.DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(Config.REPLAY_SIZE)
        
        if os.path.exists(self.model_file):
            print("Loading existing model from", self.model_file)
            self.model.load_state_dict(torch.load(self.model_file))
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            print("Creating new model")

    def update_epsilon(self):
        self.epsilon = max(
            Config.FINAL_EPSILON,
            self.epsilon - (Config.INITIAL_EPSILON - Config.FINAL_EPSILON) / Config.EPSILON_DECAY_STEPS
        )

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                q_values = self.model(state)
            action = q_values.argmax().item()
        
        self.update_epsilon()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.push(state, one_hot_action, reward, next_state, done)

    def train_network(self, batch_size: int, num_step: int):
        if len(self.replay_buffer) < batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(batch_size)

        current_q_values = self.model(state_batch)
        current_q_value = current_q_values.gather(1, 
            action_batch.argmax(dim=1, keepdim=True)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward_batch + (1 - done_batch) * Config.GAMMA * next_q_value

        loss = F.mse_loss(current_q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if num_step % Config.TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()
            if num_step % 100 == 0:  
                print(f"Step {num_step}, Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)
        print(f"Model saved to {self.model_file}")
