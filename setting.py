DQN_model_path = "model"
DQN_log_path = "logs/"
WIDTH = 96
HEIGHT = 88
#window_size = (320, 100, 704, 452)  # 384,352  192,176 96,88 48,44 24,22
# station window_size

blood_window = (546, 360, 1320, 925)
# used to get boss and self blood
# window_size = (0, 0, 1920, 1120)  # 全屏
# boss_blood_window = (560, 938, 1560, 955)  # Boss 血条区域
# self_blood_window = (198, 116, 455, 126)  # 玩家血条区域
# self_stamina_window = (198, 150, 420, 162)  # 玩家体力条区域

# wanghao
window_size = (0, 0, 1440, 810)  # 左上角
boss_blood_window = (420, 708, 1168, 712)  # Boss 血条区域
self_blood_window = (149, 91, 342, 97)  # 玩家血条区域
self_stamina_window = (149, 115, 315, 123)  # 玩家体力条区域

action_size = 13
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

paused = True
# used to stop training
