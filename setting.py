DQN_model_path = "model"
DQN_log_path = "logs/"

PPO_log_path = "logs/"
PPO_model_path = "model"

WIDTH = 192
HEIGHT = 108
# window_size = (320, 100, 704, 452)  # 384,352  192,176 96,88 48,44 24,22
# station window_size

# blood_window = (60, 91, 280, 562)
# used to get boss and self blood
window_size = (0, 0, 1920, 1080)  # 全屏
boss_blood_window = (567, 928, 1557, 943)  # Boss 血条区域
self_blood_window = (197, 105, 467, 119)  # 玩家血条区域
self_stamina_window = (204, 139, 425, 153)  # 玩家体力条区域

action_size = 5
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

paused = True
# used to stop training
