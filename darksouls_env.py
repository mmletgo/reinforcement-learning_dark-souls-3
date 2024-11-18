import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from utility import pause_game, gamestatus
from reward_fc import action_judge
from setting import WIDTH, HEIGHT, window_size, self_blood_window, boss_blood_window, self_stamina_window, action_size
import grabscreen

class DarkSoulsBossEnv(gym.Env):
    def __init__(self, env_name='train_env'):
        super(DarkSoulsBossEnv, self).__init__()
        self.env_name = env_name
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8)
        self.state = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self.prev_state = self.state.copy()
        self.emergence_break = 0
        self.stop = 0
        self.paused = False

        self.gamestatus = gamestatus()
        self.episode_start_time = None
        self.prev_action = None
        self.in_combo = False
        self.combo_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            self.gamestatus.reset()
            self.gamestatus.restart()
            time.sleep(0.1)

            status, self_blood, self_stamina, boss_blood = self.gamestatus.get_status_info()
            status = status.astype(np.uint8).reshape((HEIGHT, WIDTH, 1))
            self.state = status

            self.self_blood = self_blood
            self.next_self_blood = self_blood
            self.self_stamina = self_stamina
            
            self.next_self_stamina = self_stamina
            self.boss_blood = boss_blood
            self.next_boss_blood = boss_blood

            print(f'reset-> {self.env_name} | self blood: {self_blood}, boss blood: {boss_blood}, self stamina: {self_stamina}')

            if boss_blood < 50:
                print(f"Boss blood {boss_blood} less than 50, discarding this episode and resetting...")
                continue 
            else:
                break

        self.prev_state = self.state.copy()
        self.emergence_break = 0
        self.stop = 0
        self.episode_start_time = time.time()  
        self.prev_action = None
        self.in_combo = False
        self.combo_count = 0

        return self.state, {}

    def step(self, action):
        self.gamestatus.take_action(action)
        time.sleep(0.5)
        status, next_self_blood, next_self_stamina, next_boss_blood = self.gamestatus.get_status_info()
        print(f'step-> {self.env_name} | self blood: {next_self_blood}, boss blood: {next_boss_blood}, self stamina: {next_self_stamina}')
        status = status.astype(np.uint8).reshape((HEIGHT, WIDTH, 1))
        next_state = status

        prev_self_blood = self.self_blood
        prev_self_stamina = self.self_stamina
        prev_boss_blood = self.boss_blood
        
        self.self_blood = next_self_blood
        self.self_stamina = next_self_stamina
        self.boss_blood = next_boss_blood
        
        try:
            reward, done, self.stop, self.emergence_break, self.in_combo, self.combo_count = self.gamestatus.action_judge(
                prev_self_blood, self.self_blood,
                prev_self_stamina, self.self_stamina,
                prev_boss_blood, self.boss_blood,
                action, self.prev_action, self.stop, self.emergence_break, self.in_combo, self.combo_count
            )
            print(f"Reward: {reward}, Done: {done}, Stop: {self.stop}, Emergence Break: {self.emergence_break}, "
                f"In Combo: {self.in_combo}, Combo Count: {self.combo_count}")
        except ValueError as e:
            print(f"Error in action_judge: {e}")
            done = True
            reward = -1000

        self.prev_action = action
        
        if self.emergence_break == 100:
            print(f"Emergency break triggered: {self.env_name}")
            done = True

        self.state = next_state
        self.paused = pause_game(self.paused)
        if self.paused:
            time.sleep(1)

        truncated = False
        info = {
            'boss_blood': self.boss_blood
        }

        print(f'finish step: {self.env_name}')

        return self.state, reward, done, truncated, info


