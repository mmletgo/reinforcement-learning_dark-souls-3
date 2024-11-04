from getkeys import key_check
import time
import directkeys
import cv2
from grabscreen import grab_screen
from setting import WIDTH, HEIGHT, window_size, self_blood_window, boss_blood_window, self_stamina_window
import numpy as np


def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused


class gamestatus:

    def __init__(self):
        self.self_blood = 0
        self.boss_blood = 0
        self.action_dict = {
            0: "nothing",
            1: "left_click",
            2: "right_click",
            3: "heavy_attack_left",
            4: "sprint_jump_roll",
            5: "run_forward",
            6: "run_backward",
            7: "run_left",
            8: "run_right",
            9: "run_forward_roll",
            10: "run_backward_roll",
            11: "run_left_roll",
            12: "run_right_roll"
        }

    def reset(self):
        self.self_blood = 0
        self.boss_blood = 0

    def self_blood_count(self,
                         color_image,
                         red_self_blood_threshold=75,
                         green_self_blood_threshold=45,
                         blue_self_blood_threshold=45):
        self_blood = 0
        average_color = np.mean(color_image, axis=0)
        for pixel in average_color:
            red_value = pixel[0]
            green_value = pixel[1]
            blue_value = pixel[2]
            if red_value > red_self_blood_threshold and green_value < green_self_blood_threshold and blue_value < blue_self_blood_threshold:
                self_blood += 1
        return self_blood

    def boss_blood_count(self,
                         color_image,
                         red_boss_blood_threshold=80,
                         green_boss_blood_threshold=45,
                         blue_boss_blood_threshold=45):
        boss_blood = 0
        average_color = np.mean(color_image, axis=0)
        for pixel in average_color:
            red_value = pixel[0]
            green_value = pixel[1]
            blue_value = pixel[2]

            if red_value > red_boss_blood_threshold and green_value < green_boss_blood_threshold and blue_value < blue_boss_blood_threshold:
                boss_blood += 1
        return boss_blood

    def self_stamina_count(self,
                           color_image,
                           self_stamina_red=68,
                           self_stamina_green=80,
                           self_stamina_blue=68):
        self_stamina = 0
        average_color = np.mean(color_image, axis=0)
        for pixel in average_color:
            red_value = pixel[0]
            green_value = pixel[1]
            blue_value = pixel[2]

            if red_value < self_stamina_red and green_value > self_stamina_green and blue_value < self_stamina_blue:
                self_stamina += 1
        return self_stamina

    def get_status_info(self):
        screen_image = grab_screen(window_size)
        screen_image_rgb = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2RGB)
        self_screen_color = screen_image_rgb[
            self_blood_window[1]:self_blood_window[3],
            self_blood_window[0]:self_blood_window[2]]
        boss_screen_color = screen_image_rgb[
            boss_blood_window[1]:boss_blood_window[3],
            boss_blood_window[0]:boss_blood_window[2]]
        stamina_screen_color = screen_image_rgb[
            self_stamina_window[1]:self_stamina_window[3],
            self_stamina_window[0]:self_stamina_window[2]]
        boss_blood = self.boss_blood_count(boss_screen_color)
        if (boss_blood <= self.boss_blood and self.boss_blood - boss_blood
                < 100) or self.boss_blood == 0 or boss_blood == 0:
            # if self.boss_blood - boss_blood > 100 and boss_blood > 100 and self.boss_blood > 50:
            #     # cv2.imwrite("great_attack.png",
            #     #             cv2.cvtColor(boss_screen_color, cv2.COLOR_RGB2BGR))
            #     cv2.imwrite("great_attack.png", screen_image)
            #     np.save("great_attack.npy", screen_image_rgb)
            self.boss_blood = boss_blood
        # 计算血量和体力值
        self_blood = self.self_blood_count(self_screen_color)
        if self_blood <= self.self_blood or self.self_blood == 0 or self.boss_blood == 0:
            self.self_blood = self_blood
        self_stamina = self.self_stamina_count(stamina_screen_color)
        screen_gray = cv2.cvtColor(screen_image_rgb, cv2.COLOR_RGB2GRAY)
        status = cv2.resize(screen_gray, (WIDTH, HEIGHT))
        status = np.array(status, dtype=np.float32).reshape(HEIGHT, WIDTH)
        return status, self.self_blood, self_stamina, self.boss_blood

    def action_judge(self, self_blood, next_self_blood, self_stamina,
                     next_self_stamina, boss_blood, next_boss_blood, action,
                     stop, emergence_break):
        # get action reward
        # emergence_break is used to break down training
        if next_self_blood < 3:  # self dead
            if emergence_break < 2:
                reward = -1000 + next_self_blood - self_blood
                done = 1
                stop = 0
                emergence_break += 1
                return reward, done, stop, emergence_break
            else:
                reward = -1000 + next_self_blood - self_blood
                done = 1
                stop = 0
                emergence_break = 100
                return reward, done, stop, emergence_break
        elif next_boss_blood < 3:  # boss dead
            if emergence_break < 2:
                reward = 2000 + boss_blood - next_boss_blood
                done = 0
                stop = 0
                emergence_break += 1
                return reward, done, stop, emergence_break
            else:
                reward = 2000 + boss_blood - next_boss_blood
                done = 0
                stop = 0
                emergence_break = 100
                return reward, done, stop, emergence_break
        else:
            self_blood_reward = 0
            boss_blood_reward = 0
            # print(next_self_blood - self_blood)
            # print(next_boss_blood - boss_blood)
            if next_self_blood - self_blood < -7:
                if stop == 0:
                    self_blood_reward = next_self_blood - self_blood
                    stop = 1
                    # 防止连续取帧时一直计算掉血
            else:
                stop = 0
            if next_boss_blood - boss_blood <= -3:
                boss_blood_reward = boss_blood - next_boss_blood
            # print("self_blood_reward:    ",self_blood_reward)
            # print("boss_blood_reward:    ",boss_blood_reward)
            reward = self_blood_reward + boss_blood_reward
            if next_boss_blood == boss_blood and action != 0:
                reward -= 1
            done = 0
            emergence_break = 0
            return reward, done, stop, emergence_break

    def take_action(self, action):
        if action == 0:  # n_choose
            pass
        elif action == 1:  # 左击
            directkeys.left_click()
        elif action == 2:  # 右击（盾）
            directkeys.right_click()
        elif action == 3:  # 重击
            directkeys.heavy_attack_left()
        elif action == 4:  # 向后闪避，没加翻滚
            directkeys.sprint_jump_roll()
        elif action == 5:  # 往前走w
            directkeys.run_forward()
            time.sleep(1)
            directkeys.stop_forward()
        elif action == 6:  # 往后走s
            directkeys.run_backward()
            time.sleep(1)
            directkeys.stop_backward()
        elif action == 7:  # 往左走a
            directkeys.run_left()
            time.sleep(1)
            directkeys.stop_left()
        elif action == 8:  # 往右走d
            directkeys.run_right()
            time.sleep(1)
            directkeys.stop_right()
        elif action == 9:  # 往前翻滚w
            directkeys.run_forward()
            directkeys.sprint_jump_roll()
            time.sleep(1)
            directkeys.stop_forward()
        elif action == 10:  # 往后翻滚s
            directkeys.run_backward()
            directkeys.sprint_jump_roll()
            time.sleep(1)
            directkeys.stop_backward()
        elif action == 11:  # 往左翻滚a
            directkeys.run_left()
            directkeys.sprint_jump_roll()
            time.sleep(1)
            directkeys.stop_left()
        elif action == 12:  # 往右翻滚d
            directkeys.run_right()
            directkeys.sprint_jump_roll()
            time.sleep(1)
            directkeys.stop_right()

    def restart(self):
        while True:
            time.sleep(1)
            status, self_blood, self_stamina, boss_blood = self.get_status_info(
            )
            print("self_blood: ", self_blood, "self_stamina: ", self_stamina,
                  "boss_blood: ", boss_blood)
            if self_blood > 200 and self_stamina > 100:
                break
        time.sleep(2)
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
        time.sleep(5.5)
        directkeys.stop_forward()
        time.sleep(0.2)
        directkeys.reset_camera()
        print("restart a new episode")
