import gymnasium as gym
from stable_baselines3 import PPO
from darksouls_env import DarkSoulsBossEnv
import numpy as np
import time
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from tqdm import tqdm
import os
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor    

import csv

class LossLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossLoggingCallback, self).__init__(verbose)
        self.log_file = open('training_metrics.csv', 'w')
        self.log_file.write('timesteps,total_reward,loss,value_loss,policy_loss,entropy_loss\n')
        
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        logs = self.logger.name_to_value
        timesteps = self.num_timesteps
        loss = logs.get('train/loss', None)
        value_loss = logs.get('train/value_loss', None)
        policy_loss = logs.get('train/policy_loss', None)
        entropy_loss = logs.get('train/entropy_loss', None)
        self.log_file.write(f'{timesteps},,{loss},{value_loss},{policy_loss},{entropy_loss}\n')
        self.log_file.flush()
        
    def _on_training_end(self):
        self.log_file.close()

class StepLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StepLoggingCallback, self).__init__(verbose)
        self.log_file = open('step_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['timesteps', 'episode', 'reward', 'action', 'done', 'observation_mean', 'loss'])
        self.episode_num = 0

    def _on_step(self) -> bool:
        timesteps = self.num_timesteps
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0])
        dones = self.locals.get('dones', [False])
        actions = self.locals.get('actions', [None])
        observations = self.locals.get('new_obs', [None])

        reward = rewards[0]
        done = dones[0]
        action = actions[0]
        observation = observations[0]
        observation_mean = np.mean(observation) if observation is not None else None

        logs = self.logger.name_to_value
        loss = logs.get('train/loss', None)

        self.csv_writer.writerow([timesteps, self.episode_num, reward, action, done, observation_mean, loss])

        if done:
            self.episode_num += 1

        return True

    def _on_training_end(self):
        self.log_file.close()

class StepPrintCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StepPrintCallback, self).__init__(verbose)
        self.episode_step = 0

    def _on_step(self) -> bool:
        self.episode_step += 1
        print(f"Global Step {self.num_timesteps}, Episode Step {self.episode_step} ended.")
        dones = self.locals.get('dones', None)
        if dones is not None and any(dones):
            print(f"Episode ended after {self.episode_step} steps.")
            self.episode_step = 0

        return True

class TQDMProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps, desc="training progress")

    def _on_step(self) -> bool:
        self.progress_bar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.progress_bar.close()


class BossHealthMonitorCallback(BaseCallback):
    def __init__(self, threshold=20, consecutive_steps=10, save_path='./models/', verbose=1):
        super(BossHealthMonitorCallback, self).__init__(verbose)
        self.threshold = threshold
        self.consecutive_steps = consecutive_steps
        self.save_path = save_path
        self.counter = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            boss_blood = info.get('boss_blood', None)
            if boss_blood is not None:
                if boss_blood < self.threshold:
                    self.counter += 1
                    if self.verbose > 0:
                        print(f"Boss hp lower than {self.threshold}，times: {self.counter}")
                    if self.counter >= self.consecutive_steps:
                        if self.verbose > 0:
                            print(f"Boss hp lower than {self.threshold} times: {self.consecutive_steps}, saved model")
                        model_path = os.path.join(self.save_path, 'ppo_darksouls_boss_final')
                        self.model.save(model_path)
                        if self.verbose > 0:
                            print(f"model saved to  {model_path}")
                        return False
                else:
                    if self.counter != 0 and self.verbose > 0:
                        print(f"Boss hp recovered，times: {self.counter}")
                    self.counter = 0
        return True


def main():
    env = DarkSoulsBossEnv(env_name='train_env')
    env = Monitor(env)
    checkpoint_dir = './models/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('ppo_darksouls')]
    print('checkpoints:', checkpoints)
    if checkpoints:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        print('last checkpoints:', latest_checkpoint)
        try:
            model = PPO.load(latest_checkpoint, env=env)
            print(f"Model loaded from {latest_checkpoint}")
        except Exception as e:
            model = PPO(
                'CnnPolicy',
                env,
                verbose=1,
                tensorboard_log="./ppo_darksouls_tensorboard/",
                learning_rate=1e-4,
                n_steps=9600,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                device='cuda'
            )
            print("No compatible checkpoint found. Starting training from scratch.")
    else:
        model = PPO(
            'CnnPolicy',
            env,
            verbose=1,
            tensorboard_log="./ppo_darksouls_tensorboard/",
            learning_rate=1e-4,
            n_steps=64,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            device='cuda',
            policy_kwargs=dict(normalize_images=False)
        )
        print("No checkpoint found. Starting training from scratch.")

    checkpoint_callback = CheckpointCallback(save_freq=500, save_path=checkpoint_dir,
                                             name_prefix='ppo_darksouls')
    progress_bar_callback = TQDMProgressBarCallback(total_timesteps=1000000)
    step_print_callback = StepPrintCallback()
    boss_health_callback = BossHealthMonitorCallback(threshold=20, consecutive_steps=10, save_path=checkpoint_dir)
    loss_logging_callback = LossLoggingCallback()
    step_logging_callback = StepLoggingCallback()


    print("Starting or resuming model training...")
    model.learn(total_timesteps=1000000, tb_log_name="ppo_darksouls",
                callback=[checkpoint_callback, progress_bar_callback, step_print_callback, boss_health_callback,loss_logging_callback,step_logging_callback  ])
    print("Training completed.")
    model.save("ppo_darksouls_boss")
    print("Model saved as 'ppo_darksouls_boss'.")
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
