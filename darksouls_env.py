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

        # Define observation space: Image data (grayscale)
        #self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(HEIGHT, WIDTH, 1), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8)



        # Initialize state
        self.state = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self.prev_state = self.state.copy()
        self.emergence_break = 0
        self.stop = 0
        self.paused = False

        # Initialize game status
        self.gamestatus = gamestatus()
        self.episode_start_time = None  # Add this line to record episode start time
        self.prev_action = None
        self.in_combo = False
        self.combo_count = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment and restart the battle.
        """
        super().reset(seed=seed)
        while True:
            self.gamestatus.reset()
            self.gamestatus.restart()  # Restart the game or battle
            time.sleep(8)  # Wait for the game to load and restart

            # 获取初始状态
            status, self_blood, self_stamina, boss_blood = self.gamestatus.get_status_info()
            # 对图像进行归一化处理
            status = status.astype(np.uint8).reshape((HEIGHT, WIDTH, 1))
            self.state = status

            # 保存血量和耐力信息
            self.self_blood = self_blood
            self.next_self_blood = self_blood
            self.self_stamina = self_stamina
            
            self.next_self_stamina = self_stamina
            self.boss_blood = boss_blood
            self.next_boss_blood = boss_blood

            print(f'reset-> {self.env_name} | self blood: {self_blood}, boss blood: {boss_blood}, self stamina: {self_stamina}')

            if boss_blood < 50:
                print(f"Boss blood {boss_blood} less than 50, discarding this episode and resetting...")
                continue  # 重新开始 reset
            else:
                break  # 跳出循环，开始新的 Episode

        self.prev_state = self.state.copy()
        self.emergence_break = 0
        self.stop = 0
        self.episode_start_time = time.time()  # Record the start time of the episode
        self.prev_action = None
        self.in_combo = False
        self.combo_count = 0

        return self.state, {}

    def step(self, action):
        self.gamestatus.take_action(action)
        time.sleep(0.5)  # Delay for action execution

        # Get the next state
        status, next_self_blood, next_self_stamina, next_boss_blood = self.gamestatus.get_status_info()
        print(f'step-> {self.env_name} | self blood: {next_self_blood}, boss blood: {next_boss_blood}, self stamina: {next_self_stamina}')

        # Ensure status is uint8 and has channel dimension
        status = status.astype(np.uint8).reshape((HEIGHT, WIDTH, 1))
        next_state = status

        # Store previous blood and stamina
        prev_self_blood = self.self_blood
        prev_self_stamina = self.self_stamina
        prev_boss_blood = self.boss_blood

        # Update current blood and stamina
        self.self_blood = next_self_blood
        self.self_stamina = next_self_stamina
        self.boss_blood = next_boss_blood

        # Calculate reward, capture additional return values
        try:
            reward, done, self.stop, self.emergence_break, self.in_combo, self.combo_count = self.gamestatus.action_judge(
                prev_self_blood, self.self_blood,
                prev_self_stamina, self.self_stamina,
                prev_boss_blood, self.boss_blood,
                action, self.prev_action, self.stop, self.emergence_break,self.in_combo, self.combo_count
            )
            print(f"Reward: {reward}, Done: {done}, Stop: {self.stop}, Emergence Break: {self.emergence_break}, "
                f"In Combo: {self.in_combo}, Combo Count: {self.combo_count}")
        except ValueError as e:
            print(f"Error in action_judge: {e}")
            done = True
            reward = -1000

        # **修正这里，将 self.action 替换为 action**
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

        return self.state, reward, done, truncated, info

