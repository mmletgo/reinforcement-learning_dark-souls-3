# -*- coding: utf-8 -*-
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
from model import DQN
from restart import restart
from utility import pause_game, self_blood_count, boss_blood_count, take_action
from reward_fc import action_judge
from setting import DQN_model_path, DQN_log_path, WIDTH, HEIGHT, window_size, blood_window, action_size, paused

if __name__ == '__main__':
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    # DQN init
    paused = pause_game(paused)
    # paused at the begin
    screen_gray = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_BGR2GRAY)
    blood_window_gray = cv2.cvtColor(grab_screen(blood_window),
                                     cv2.COLOR_BGR2GRAY)
    # collect station gray graph
    station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
    # change graph to WIDTH * HEIGHT for station input
    boss_blood = boss_blood_count(blood_window_gray)
    self_blood = self_blood_count(blood_window_gray)
    # count init blood
    target_step = 0
    # used to update target Q network
    done = 0
    total_reward = 0
    stop = 0
    # 用于防止连续帧重复计算reward
    last_time = time.time()
    emergence_break = 0
    while True:
        station = np.array(station).reshape(-1, HEIGHT, WIDTH, 1)[0]
        # reshape station for tf input placeholder
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        # get the action by state
        action = agent.Choose_Action(station)
        take_action(action)
        # take station then the station change
        screen_gray = cv2.cvtColor(grab_screen(window_size),
                                   cv2.COLOR_BGR2GRAY)
        # collect station gray graph
        blood_window_gray = cv2.cvtColor(grab_screen(blood_window),
                                         cv2.COLOR_BGR2GRAY)
        # collect blood gray graph for count self and boss blood
        next_station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
        next_station = np.array(next_station).reshape(-1, HEIGHT, WIDTH, 1)[0]
        station = next_station
        next_boss_blood = boss_blood_count(blood_window_gray)
        next_self_blood = self_blood_count(blood_window_gray)
        reward, done, stop, emergence_break = action_judge(
            boss_blood, next_boss_blood, self_blood, next_self_blood, stop,
            emergence_break)
        # get action reward
        if emergence_break == 100:
            # emergence break , save model and paused
            # 遇到紧急情况，保存数据，并且暂停
            print("emergence_break")
            agent.save_model()
            paused = True
        keys = key_check()
        paused = pause_game(paused)
        if 'G' in keys:
            print('stop testing DQN')
            break
        if done == 1:
            restart()
