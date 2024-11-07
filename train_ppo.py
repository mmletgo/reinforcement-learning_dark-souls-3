# train_ppo.py
import gymnasium as gym
from stable_baselines3 import PPO
from darksouls_env import DarkSoulsBossEnv
import numpy as np
import time
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from tqdm import tqdm
import os
import torch

class StepPrintCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StepPrintCallback, self).__init__(verbose)
        self.episode_step = 0  # 用于记录当前 episode 的 step 数

    def _on_step(self) -> bool:
        self.episode_step += 1
        print(f"Global Step {self.num_timesteps}, Episode Step {self.episode_step} ended.")

        # 检查当前 episode 是否结束
        dones = self.locals.get('dones', None)
        if dones is not None and any(dones):
            print(f"Episode ended after {self.episode_step} steps.")
            self.episode_step = 0  # 重置 episode step 计数器

        return True

class TQDMProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps, desc="训练进度")

    def _on_step(self) -> bool:
        self.progress_bar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.progress_bar.close()

def main():
    # 创建训练环境
    env = DarkSoulsBossEnv(env_name='train_env')
    
    # 检查是否存在检查点
    checkpoint_dir = './models/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('ppo_darksouls')]
    if checkpoints:
        # 按照训练步数排序，加载最新的检查点
        checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        model = PPO.load(latest_checkpoint, env=env)
        print(f"Model loaded from {latest_checkpoint}")
    else:
        # 如果没有检查点，创建新模型
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log="./ppo_darksouls_tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            device='cuda'
        )
        print("No checkpoint found. Starting training from scratch.")

    # 设置检查点回调和自定义回调
    checkpoint_callback = CheckpointCallback(save_freq=499, save_path=checkpoint_dir,
                                             name_prefix='ppo_darksouls')
    progress_bar_callback = TQDMProgressBarCallback(total_timesteps=1000000)
    step_print_callback = StepPrintCallback()

    # 开始或继续训练
    print("Starting or resuming model training...")
    model.learn(total_timesteps=1000000, tb_log_name="ppo_darksouls",
                callback=[checkpoint_callback, progress_bar_callback, step_print_callback])
    print("Training completed.")

    # 保存最终模型
    model.save("ppo_darksouls_boss")
    print("Model saved as 'ppo_darksouls_boss'.")

    # 测试模型
    test_model(env, model)

def test_model(env, model, episodes=10):
    """
    Test the trained model.
    """
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            print(f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
            if done:
                print(f"Episode {episode + 1} ended with total reward: {total_reward}")
                break

if __name__ == "__main__":
    main()
