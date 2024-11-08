import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard库


class PolicyNetwork(nn.Module):

    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 增加额外的 MLP 层
        self.fc1 = nn.Linear(64 * 10 * 20, 512)
        self.fc2 = nn.Linear(512, 256)  # 增加一层
        self.fc3 = nn.Linear(256, 128)  # 增加另一层
        # Dropout 层
        self.dropout = nn.Dropout(p=0.3)
        # 输出层
        self.action_layer = nn.Linear(128, action_dim)
        # 初始化最后一层的权重，使输出初始概率接近均等
        nn.init.constant_(self.action_layer.weight, 0.0)  # 将权重初始化为 0
        nn.init.constant_(self.action_layer.bias, 0.0)  # 将偏置初始化为 0

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        # 经过 MLP 层和 Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 第一个 Dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # 第二个 Dropout
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)  # 第三个 Dropout
        logits = self.action_layer(x)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]  # 归一化，防止数值过大
        action_probs = torch.softmax(logits, dim=-1) + 1e-8  # 增加一个小常数以确保非零
        return action_probs

    def act(self, state):
        action_probs = self.forward(state)
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)


class ValueNetwork(nn.Module):

    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 增加额外的 MLP 层
        self.fc1 = nn.Linear(64 * 10 * 20, 512)
        self.fc2 = nn.Linear(512, 256)  # 增加一层
        self.fc3 = nn.Linear(256, 128)  # 增加另一层
        # Dropout 层
        self.dropout = nn.Dropout(p=0.3)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        # 经过 MLP 层和 Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 第一个 Dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # 第二个 Dropout
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)  # 第三个 Dropout
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
        self.policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.policy_optimizer, mode='min', factor=0.5, patience=10)
        self.value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.value_optimizer, mode='min', factor=0.5, patience=10)

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
        # 切换到训练模式
        self.policy_net.train()
        self.value_net.train()
        with torch.no_grad():
            states_np = np.array(self.states, dtype=np.float32)
            states = torch.tensor(states_np).unsqueeze(1).to(self.device).detach()
            print(states.shape)
            actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device).detach()
            rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32).to(self.device).detach()
            logprobs = torch.tensor(self.logprobs, dtype=torch.float32).to(self.device).detach()

            self.writer.add_scalar("Loss/Rewards", rewards.sum().item(), self.update_count)

            # max_reward = 200
            # min_reward = -100
            # rewards = (rewards - min_reward) / (max_reward - min_reward)

            values = self.value_net(states).squeeze()

            advantages = torch.zeros_like(rewards).to(self.device)
            last_gae_lambda = 0
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    delta = rewards[i] - values[i]
                    advantages[i] = last_gae_lambda = delta
                else:
                    delta = rewards[i] + self.gamma * values[i + 1] - values[i]
                    advantages[i] = last_gae_lambda = delta + self.gamma * 0.95 * last_gae_lambda
            returns = advantages + values

            # normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            normalized_advantages = advantages

        # 使用每个采样的策略和价值网络更新
        for epoch in range(self.K_epochs):
            action_probs = self.policy_net(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_logprobs = action_dist.log_prob(actions)
            dist_entropy = action_dist.entropy()
            new_values = self.value_net(states).squeeze()

            ratios = torch.exp(new_logprobs - logprobs)
            surr1 = ratios * normalized_advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * normalized_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            true_policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()

            # 检查是否出现 NaN，如果有则跳过本次更新
            if torch.isnan(true_policy_loss) or torch.isnan(new_values).any():
                print("NaN detected in policy loss or new values. Skipping update.")
                return  # 终止本次更新

            self.policy_optimizer.zero_grad()
            true_policy_loss.backward()
            # Clip the gradient to stabilize training
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.1)
            self.policy_optimizer.step()

            value_loss_unclipped = (new_values - returns).pow(2) / 2
            values_clipped = values + torch.clamp(new_values - values, - self.eps_clip, self.eps_clip)
            value_loss_clipped = (values_clipped - returns).pow(2) / 2
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # 检查是否出现 NaN，如果有则跳过价值网络更新
            if torch.isnan(value_loss):
                print("NaN detected in value loss. Skipping value update.")
                continue  # 跳过本次价值网络更新

            self.value_optimizer.zero_grad()
            value_loss.backward()
            # Clip the gradient to stabilize training
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()

            # 将损失写入TensorBoard
            self.writer.add_scalar("Loss/Policy_Loss", policy_loss.item(),
                                   self.update_count * self.K_epochs + epoch)
            self.writer.add_scalar("Loss/Value_Loss", value_loss.item(),
                                   self.update_count * self.K_epochs + epoch)
            self.writer.add_scalar("Loss/Entropy", 0.01 * dist_entropy.mean().item(), self.update_count * self.K_epochs + epoch)
        # 增加更新计数
        self.update_count += 1
        self.policy_scheduler.step(policy_loss)
        self.value_scheduler.step(value_loss)

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
