# -*- coding: utf-8 -*-
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from model import DQN
from restart import restart
from utility import pause_game, self_blood_count, boss_blood_count, take_action
from reward_fc import action_judge
from setting import DQN_model_path, DQN_log_path, WIDTH, HEIGHT, window_size, blood_window, action_size, paused

EPISODES = 3000
big_BATCH_SIZE = 16
UPDATE_STEP = 50
# times that evaluate the network
num_step = 0
# used to save log graph
target_step = 0
# used to update target Q network

if __name__ == '__main__':
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    # DQN init
    paused = pause_game(paused)
    # paused at the begin
    emergence_break = 0
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    for episode in range(EPISODES):
        screen_gray = cv2.cvtColor(grab_screen(window_size),
                                   cv2.COLOR_BGR2GRAY)
        # collect station gray graph
        blood_window_gray = cv2.cvtColor(grab_screen(blood_window),
                                         cv2.COLOR_BGR2GRAY)
        # collect blood gray graph for count self and boss blood
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
        while True:
            station = np.array(station).reshape(-1, HEIGHT, WIDTH, 1)[0]
            # reshape station for tf input placeholder
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            target_step += 1
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
            next_station = np.array(next_station).reshape(
                -1, HEIGHT, WIDTH, 1)[0]
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
            agent.Store_Data(station, action, reward, next_station, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                # save loss graph
                # print('train')
                agent.Train_Network(big_BATCH_SIZE, num_step)
            if target_step % UPDATE_STEP == 0:
                agent.Update_Target_Network()
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
