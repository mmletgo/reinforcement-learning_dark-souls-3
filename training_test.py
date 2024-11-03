# -*- coding: utf-8 -*-
from grabscreen import grab_screen
import cv2
from restart import restart
from utility import pause_game, self_blood_count_grey, boss_blood_count, self_stamina_count_grey, take_action
from setting import window_size, self_blood_window, boss_blood_window, self_stamina_window, paused

if __name__ == '__main__':
    # DQN init
    paused = pause_game(paused)
    # paused at the begin
    emergence_break = 0
    # emergence_break is used to break down training
    restart()
    while True:
        # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
        screen_image = grab_screen(window_size)

        self_screen_color = cv2.cvtColor(
            screen_image[self_blood_window[1]:self_blood_window[3],
                         self_blood_window[0]:self_blood_window[2]],
            cv2.COLOR_BGR2GRAY)
        boss_screen_color = screen_image[
            boss_blood_window[1]:boss_blood_window[3],
            boss_blood_window[0]:boss_blood_window[2]]
        stamina_screen_color = cv2.cvtColor(
            screen_image[self_stamina_window[1]:self_stamina_window[3],
                         self_stamina_window[0]:self_stamina_window[2]],
            cv2.COLOR_BGR2GRAY)
        # 计算血量和体力值
        self_blood = self_blood_count_grey(self_screen_color)
        boss_blood = boss_blood_count(boss_screen_color)
        self_stamina = self_stamina_count_grey(stamina_screen_color)
        print(
            f"self_blood:{self_blood}, boss_blood:{boss_blood}, self_stamina:{self_stamina}"
        )
        take_action(1)
        if self_blood == 0:
            restart()
