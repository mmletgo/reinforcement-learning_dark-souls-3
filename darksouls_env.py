# darksouls_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from utility import pause_game, gamestatus
from reward_fc import action_judge
from setting import WIDTH, HEIGHT, window_size, self_blood_window, boss_blood_window, self_stamina_window, action_size
import grabscreen

class DarkSoulsBossEnv(gym.Env):
    """
    Custom Gym environment for training a PPO model to automatically fight a boss.
    """
    def __init__(self, env_name='train_env'):
        super(DarkSoulsBossEnv, self).__init__()
        self.env_name = env_name  # Environment name identifier

        # Define action space: 13 discrete actions
        self.action_space = spaces.Discrete(action_size)

        # Define observation space: Player HP, Boss HP, Player Stamina
        # Adjust the ranges according to your game mechanics
        self.observation_space = spaces.Box(low=0, high=500, shape=(3,), dtype=np.float32)

        # Initialize state
        self.state = np.array([500, 500, 500], dtype=np.float32)
        self.prev_state = self.state.copy()
        self.emergence_break = 0
        self.stop = 0
        self.paused = False

        # Initialize game status
        self.gamestatus = gamestatus()
        self.episode_start_time = None  # Add this line to record episode start time
        self.prev_action = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment and restart the battle.
        """
        super().reset(seed=seed)
        while True:
            self.gamestatus.reset()
            self.gamestatus.restart()  # Restart the game or battle
            time.sleep(8)  # Wait for the game to load and restart

            # Capture initial state
            status, self_blood, self_stamina, boss_blood = self.gamestatus.get_status_info()
            print(f'reset-> {self.env_name} | self blood: {self_blood}, boss blood: {boss_blood}, self stamina: {self_stamina}')
            
            if boss_blood < 50:
                print(f"Boss blood {boss_blood} less than 50, discarding this episode and resetting...")
                continue  # 重新开始 reset
            else:
                break  # 跳出循环，开始新的 Episode

        self.state = np.array([self_blood, boss_blood, self_stamina], dtype=np.float32)
        self.prev_state = self.state.copy()
        self.emergence_break = 0
        self.stop = 0
        self.episode_start_time = time.time()  # Record the start time of the episode
        self.prev_action = None

        return self.state, {}

    def step(self, action):
        self.gamestatus.take_action(action)
        time.sleep(0.5)  # Delay for action execution

        # Get the next state
        status, next_self_blood, next_self_stamina, next_boss_blood = self.gamestatus.get_status_info()
        print(f'step-> {self.env_name} | self blood: {next_self_blood}, boss blood: {next_boss_blood}, self stamina: {next_self_stamina}')

        next_state = np.array([next_self_blood, next_boss_blood, next_self_stamina], dtype=np.float32)

        # Calculate reward
        reward, done, self.stop, self.emergence_break = self.gamestatus.action_judge(
            self.state[0], next_self_blood,
            self.state[2], next_self_stamina,
            self.state[1], next_boss_blood,
            action, self.prev_action,self.stop, self.emergence_break
        )
        self.prev_action = action

        # Check for emergency break
        if self.emergence_break == 100:
            print(f"Emergency break triggered: {self.env_name}")
            done = True

        # Update state
        self.state = next_state

        # Handle pause
        self.paused = pause_game(self.paused)
        if self.paused:
            time.sleep(1)  # Wait while paused

        truncated = False
        info = {}

        print(f'finish step: {self.env_name}')

        return next_state, reward, done, truncated, info
