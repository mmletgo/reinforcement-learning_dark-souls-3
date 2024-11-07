# -*- coding: utf-8 -*-
import directkeys
import time
import cv2
from grabscreen import grab_screen
from setting import window_size, self_blood_window, self_stamina_window
from utility import self_blood_count_grey, self_stamina_count_grey


def restart():
    while True:
        time.sleep(1)
        screen_image = grab_screen(window_size)
        self_screen_color = cv2.cvtColor(
            screen_image[self_blood_window[1]:self_blood_window[3],
                         self_blood_window[0]:self_blood_window[2]],
            cv2.COLOR_BGR2GRAY)
        stamina_screen_color = cv2.cvtColor(
            screen_image[self_stamina_window[1]:self_stamina_window[3],
                         self_stamina_window[0]:self_stamina_window[2]],
            cv2.COLOR_BGR2GRAY)
        self_blood = self_blood_count_grey(self_screen_color)
        self_stamina = self_stamina_count_grey(stamina_screen_color)
        if self_blood > 200 and self_stamina > 200:
            break
    time.sleep(1)
    print("dead,restart")
    directkeys.teleport()
    time.sleep(0.2)
    directkeys.run_forward()
    time.sleep(1)
    directkeys.stop_forward()
    time.sleep(0.2)
    directkeys.action()
    time.sleep(4)
    directkeys.run_forward()
    time.sleep(6.5)
    directkeys.stop_forward()
    time.sleep(0.2)
    # directkeys.action()
    # time.sleep(0.2)
    # directkeys.sprint_jump_roll()
    # time.sleep(0.2)
    directkeys.reset_camera()
    print("restart a new episode")


if __name__ == "__main__":
    restart()
