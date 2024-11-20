# -*- coding: utf-8 -*-
import numpy as np
from torch.distributed.pipelining.schedules import WEIGHT

from grabscreen import grab_screen
import cv2
import time
from model_dqn1 import DQNAgent as DQN
from restart import restart
from utility import pause_game, self_blood_count, boss_blood_count,self_stamina_count, take_action
from reward_fc import action_judge
from setting import DQN_model_path, DQN_log_path, WIDTH, HEIGHT, window_size, self_blood_window,boss_blood_window,self_stamina_window, action_size, paused, blood_window

EPISODES = 3000
big_BATCH_SIZE = 16
UPDATE_STEP = 50
# times that evaluate the network
num_step = 0
# used to save log graph
target_step = 0
# used to update target Q network

if __name__ == '__ma_in__':
    time.sleep(2)
    screen_shot = grab_screen(window_size)
    station = cv2.resize(screen_shot, (WIDTH, HEIGHT))
    cv2.imwrite(f"processed_screen_.png", screen_shot)
    cv2.imshow('Processed Image', screen_shot)
    while True:
        time.sleep(1)
if __name__ == '__main__':
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    # DQN init
    paused = pause_game(paused)
    # paused at the beginning
    emergence_break = 0
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    for episode in range(EPISODES):
        screen_shot = grab_screen(window_size)
        screen_gray = cv2.cvtColor(screen_shot, cv2.COLOR_BGR2GRAY)
        # blood_window = grab_screen(blood_window)
        # collect station gray graph
        # blood_window_gray = cv2.cvtColor(grab_screen(blood_window),
        #                                  cv2.COLOR_BGR2GRAY)
        # collect blood gray graph for count self and boss blood
        station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
        # station = cv2.resize(screen_shot, (WIDTH, HEIGHT))
        # cv2.imshow('Processed Image', screen_shot)
        # change graph to WIDTH * HEIGHT for station input
        # boss_blood = boss_blood_count(blood_window_gray)
        # self_blood = self_blood_count(blood_window_gray)
        self_screen_color = screen_shot[self_blood_window[1]:self_blood_window[3],
                            self_blood_window[0]:self_blood_window[2]]
        boss_screen_color = screen_shot[boss_blood_window[1]:boss_blood_window[3],
                            boss_blood_window[0]:boss_blood_window[2]]
        self_blood = self_blood_count(self_screen_color)
        boss_blood = boss_blood_count(boss_screen_color)
        print(f"Player blood: {self_blood:.2f}% | Boss blood: {boss_blood:.2f}%")
        # count init blood
        target_step = 0
        # used to update target Q network
        done = 0
        total_reward = 0
        stop = 0
        # 用于防止连续帧重复计算reward
        last_time = time.time()
        while True:
            # station = np.array(station).reshape(-1, HEIGHT, WIDTH, 1)[0]
            station = station.reshape(1, 1, HEIGHT, WIDTH)
            # reshape station for tf input placeholder
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            target_step += 1
            # get the action by state
            action = agent.choose_action(station)
            take_action(action)
            # take station then the station change
            # screen_gray = cv2.cvtColor(grab_screen(window_size),
            #                            cv2.COLOR_BGR2GRAY)
            next_screen_shot = grab_screen(window_size)
            next_screen_gray = cv2.cvtColor(next_screen_shot, cv2.COLOR_BGR2GRAY)
            # blood_window = grab_screen(blood_window)
            # collect station gray graph
            # blood_window_gray = cv2.cvtColor(grab_screen(blood_window),
            #                                  cv2.COLOR_BGR2GRAY)
            # collect blood gray graph for count self and boss blood
            # next_station = cv2.resize(screen_gray, (WIDTH, HEIGHT))

            next_self_screen_color = next_screen_shot[self_blood_window[1]:self_blood_window[3],
                                self_blood_window[0]:self_blood_window[2]]
            next_boss_screen_color = next_screen_shot[boss_blood_window[1]:boss_blood_window[3],
                                boss_blood_window[0]:boss_blood_window[2]]

            next_station = cv2.resize(next_screen_gray, (WIDTH, HEIGHT))
            # next_station = np.array(next_station).reshape(-1, HEIGHT, WIDTH, 1)[0]
            next_station = next_station.reshape(1, 1, HEIGHT, WIDTH)
            # next_boss_blood = boss_blood_count(blood_window_gray)
            # next_self_blood = self_blood_count(blood_window_gray)
            next_self_blood = self_blood_count(next_self_screen_color)
            next_boss_blood = boss_blood_count(next_boss_screen_color)
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
            agent.store_data(station, action, reward, next_station, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                # save loss graph
                # print('train')
                agent.train_network(big_BATCH_SIZE, num_step)
            if target_step % UPDATE_STEP == 0:
                agent.update_target_network()
                # update target Q network
            station = next_station
            self_blood = next_self_blood
            boss_blood = next_boss_blood
            total_reward += reward
            paused = pause_game(paused)
            if done == 1:
                break
        if episode % 10 == 0:
            agent.save_model()
            # save model
        print('episode: ', episode, 'Evaluation Average Reward:',
              total_reward / target_step)
        restart()
