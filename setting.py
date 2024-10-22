DQN_model_path = "model"
DQN_log_path = "logs/"
WIDTH = 96
HEIGHT = 88
window_size = (320, 100, 704, 452)  # 384,352  192,176 96,88 48,44 24,22
# station window_size

blood_window = (60, 91, 280, 562)
# used to get boss and self blood

action_size = 5
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

paused = True
# used to stop training
