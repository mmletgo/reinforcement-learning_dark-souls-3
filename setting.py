DQN_model_path = "model"
DQN_log_path = "logs/"
WIDTH = 96
HEIGHT = 88
#window_size = (320, 100, 704, 452)  # 384,352  192,176 96,88 48,44 24,22
# station window_size

#blood_window = (60, 91, 280, 562)
# used to get boss and self blood
window_size = (0, 0, 1920, 1080)  # 全屏
boss_blood_window = (557, 899, 1557, 910)  # Boss 血条区域
self_blood_window = (197, 75, 453, 89)  # 玩家血条区域
self_stamina_window = (197, 109, 419, 123)  # 玩家体力条区域

action_size = 5
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

paused = True
# used to stop training
