import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard库


class PolicyNetwork(nn.Module):

    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 10 * 20, 512)
        self.action_layer = nn.Linear(512, action_dim)
        # 初始化最后一层的权重，使输出初始概率接近均等
        nn.init.constant_(self.action_layer.weight, 1e-8)  # 将权重初始化为 0
        nn.init.constant_(self.action_layer.bias, 1e-8)  # 将偏置初始化为 0

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.action_layer(x), dim=-1)
        return action_probs

    def act(self, state):
        action_probs = self.forward(state)
        # if torch.isnan(action_probs).any():
        #     print(
        #         "Action probabilities contain NaN values. Setting all values to 1e-8."
        #     )
        #     action_probs = torch.full_like(action_probs, 1e-8)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)


class ValueNetwork(nn.Module):

    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 10 * 20, 512)
        self.value_layer = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        state_value = self.value_layer(x)
        return state_value


class PPO:

    def __init__(self, action_dim, params, log_dir="runs/ppo_experiment"):
        self.gamma = params['gamma']
        self.lr = params['lr']
        self.eps_clip = params['eps_clip']
        self.K_epochs = params['K_epochs']
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 初始化策略网络和价值网络
        self.policy_net = PolicyNetwork(action_dim).to(self.device)
        self.value_net = ValueNetwork().to(self.device)

        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(),
                                           lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(),
                                          lr=self.lr)

        # 内部经验存储
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

        # 初始化TensorBoard的SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        self.update_count = 0  # 记录更新次数

        # 检查并加载模型
        self.load_model()

    def load_model(self):
        """检查并加载模型参数"""
        try:
            checkpoint = torch.load('ppo_model.pth', map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            print(f"Model loaded from {'ppo_model.pth'}")
        except FileNotFoundError:
            print("No pre-trained model found. Training from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}. Training from scratch.")

    def store_transition(self, state, action, logprob, reward):
        # 存储单步经验
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)

    def clear_memory(self):
        # 清空经验
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def update(self):
        states_np = np.array(self.states, dtype=np.float32)
        states = torch.tensor(states_np).unsqueeze(1).to(self.device).detach()
        print(states.shape)
        actions = torch.tensor(np.array(self.actions),
                               dtype=torch.float32).to(self.device).detach()
        rewards = torch.tensor(np.array(self.rewards),
                               dtype=torch.float32).to(self.device).detach()
        logprobs = torch.tensor(self.logprobs,
                                dtype=torch.float32).to(self.device).detach()

        # 计算折扣回报
        returns = deque()
        Gt = 0
        for reward in reversed(rewards):
            Gt = reward + self.gamma * Gt
            returns.appendleft(Gt)
        returns = list(returns)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # 使用每个采样的策略和价值网络更新
        for epoch in range(self.K_epochs):
            action_probs = self.policy_net(states)
            # if torch.isnan(action_probs).any():
            #     print(
            #         "Action probabilities contain NaN values. Ending function."
            #     )
            #     return
            action_dist = torch.distributions.Categorical(action_probs)
            new_logprobs = action_dist.log_prob(actions)
            dist_entropy = action_dist.entropy()

            ratios = torch.exp(new_logprobs - logprobs)
            state_values = self.value_net(states).squeeze()
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages

            # 计算损失
            policy_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
            policy_loss = policy_loss.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            value_loss = 0.5 * (returns - state_values)**2
            value_loss = value_loss.mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # 将损失写入TensorBoard
            self.writer.add_scalar("Loss/Policy_Loss", policy_loss.item(),
                                   self.update_count * self.K_epochs + epoch)
            self.writer.add_scalar("Loss/Value_Loss", value_loss.item(),
                                   self.update_count * self.K_epochs + epoch)

        # 增加更新计数
        self.update_count += 1

    def Choose_Action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(
            self.device)
        action, logprob = self.policy_net.act(state)
        return action, logprob

    def save_model(self, filepath='ppo_model.pth'):
        # 保存策略网络和价值网络的参数
        torch.save(
            {
                'policy_net': self.policy_net.state_dict(),
                'value_net': self.value_net.state_dict()
            }, filepath)
        print(f"Model saved to {filepath}")
