from getkeys import key_check
import time
import directkeys
import cv2
from grabscreen import grab_screen
from setting import WIDTH, HEIGHT, window_size, self_blood_window, boss_blood_window, self_stamina_window
import numpy as np
import os


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
                         red_self_blood_threshold=85,
                         green_self_blood_threshold=70,
                         blue_self_blood_threshold=70):
        self_blood = 0
        average_color = np.mean(color_image, axis=0)
        self.self_blood_row = average_color
        #print('self blood:',self_blood_row)
        for pixel in average_color:
            red_value = pixel[0]
            green_value = pixel[1]
            blue_value = pixel[2]
            if red_value > red_self_blood_threshold and green_value < green_self_blood_threshold and blue_value < blue_self_blood_threshold:
                self_blood += 1
        return self_blood

    def boss_blood_count(self,
                         color_image,
                         red_boss_blood_threshold=60,
                         green_boss_blood_threshold=40,
                         blue_boss_blood_threshold=40):
        boss_blood = 0
        average_color = np.mean(color_image, axis=0)
        self.boss_blood_row = average_color
        #print('boss blood:',boss_blood_row)
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
                           self_stamina_green=85,
                           self_stamina_blue=68):
        self_stamina = 0
        average_color = np.mean(color_image, axis=0)
        self.self_stamina_row = average_color
        #print('self stamina:',self_stamina_row)
        for pixel in average_color:
            red_value = pixel[0]
            green_value = pixel[1]
            blue_value = pixel[2]

            if red_value > self_stamina_red and green_value > self_stamina_green: #and blue_value < self_stamina_blue:
                self_stamina += 1
        return self_stamina

    def get_status_info(self):
        if not os.path.exists('data'):
            os.makedirs('data')

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

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.timestamp = timestamp  # store timestamp in self

        # Get health percentages
        self_blood = self.self_blood_count(self_screen_color)
        boss_blood = self.boss_blood_count(boss_screen_color)
        self_stamina = self.self_stamina_count(stamina_screen_color)
        # 确保返回的血量和耐力是标量
        self_blood = float(self_blood)
        self_stamina = float(self_stamina)
        boss_blood = float(boss_blood)

        # Save matrices to txt file
        # with open(f"data/matrix_{timestamp}.txt", 'w') as f:
        #     f.write(f"Timestamp: {timestamp}\n")
        #     f.write("\nSelf Blood Matrix (Row 5):\n")
        #     np.savetxt(f, self.self_blood_row, fmt='%d')
        #     f.write("\nBoss Blood Matrix (Row 5):\n")
        #     np.savetxt(f, self.boss_blood_row, fmt='%d')
        #     f.write("\nSelf Stamina Matrix (Row 5):\n")
        #     np.savetxt(f, self.self_stamina_row, fmt='%d')

        # Update blood values
        if (boss_blood <= self.boss_blood and self.boss_blood - boss_blood < 100) or self.boss_blood == 0 or boss_blood == 0:
            self.boss_blood = boss_blood

        if self_blood <= self.self_blood or self.self_blood == 0 or self.boss_blood == 0:
            self.self_blood = self_blood

        # Draw rectangles and save annotated image
        # annotated_image = screen_image_rgb.copy()
        # cv2.rectangle(annotated_image,
        #               (self_blood_window[0], self_blood_window[1]),
        #               (self_blood_window[2], self_blood_window[3]),
        #               (0, 255, 0), 2)  # Green rectangle for player's health bar

        # cv2.rectangle(annotated_image,
        #               (boss_blood_window[0], boss_blood_window[1]),
        #               (boss_blood_window[2], boss_blood_window[3]),
        #               (0, 0, 255), 2)  # Red rectangle for boss's health bar

        # cv2.rectangle(annotated_image,
        #               (self_stamina_window[0], self_stamina_window[1]),
        #               (self_stamina_window[2], self_stamina_window[3]),
        #               (255, 0, 0), 2)  # Blue rectangle for player's stamina bar

        # # Add labels to the rectangles
        # cv2.putText(annotated_image, 'Player Health',
        #             (self_blood_window[0], self_blood_window[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # cv2.putText(annotated_image, 'Boss Health',
        #             (boss_blood_window[0], boss_blood_window[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # cv2.putText(annotated_image, 'Player Stamina',
        #             (self_stamina_window[0], self_stamina_window[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # # Save the annotated image
        # cv2.imwrite(f"data/annotated_frame_{timestamp}.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Prepare the status for the environment
        screen_gray = cv2.cvtColor(screen_image_rgb, cv2.COLOR_RGB2GRAY)
        status = cv2.resize(screen_gray, (WIDTH, HEIGHT))
        status = np.array(status, dtype=np.float32).reshape(HEIGHT, WIDTH)
        status = status/255.0

        return status, self.self_blood, self_stamina, self.boss_blood

    def action_judge(self, self_blood, next_self_blood, self_stamina, next_self_stamina,
                 boss_blood, next_boss_blood, action, prev_action, stop, emergence_break,
                 in_combo, combo_count):
        # 设定体力阈值
        high_stamina_threshold = 50
        low_stamina_threshold = 20

        # 判断角色是否死亡
        if next_self_blood < 3:
            reward = -1000
            done = 1
            stop = 0
            emergence_break += 1 if emergence_break < 2 else 100
            return reward, done, stop, emergence_break, False, 0  # 重置连击状态
        # 判断 Boss 是否死亡
        elif next_boss_blood < 3:
            reward = 2000
            done = 0
            stop = 0
            emergence_break += 1 if emergence_break < 2 else 100
            return reward, done, stop, emergence_break, False, 0  # 重置连击状态
        else:
            # 初始化奖励
            self_blood_reward = 0
            boss_blood_reward = 0
            action_reward = 0
            stamina_penalty = 0
            idle_penalty = 0
            combo_reward = 0  # 用于奖励连击

            # 计算角色血量变化
            if next_self_blood < self_blood:
                self_blood_reward = (next_self_blood - self_blood) * 3  # 负值，受到伤害

            # 计算 Boss 血量变化
            boss_damage = boss_blood - next_boss_blood
            if boss_damage > 0:
                boss_blood_reward = boss_damage * 3  # 正值，给予更多奖励

            # 获取当前体力
            current_stamina = next_self_stamina

            # 动作奖励和体力管理
            if action in [1, 3]:  # 攻击动作
                if current_stamina > high_stamina_threshold:
                    action_reward = 20  # 体力充足，鼓励攻击
                elif current_stamina < low_stamina_threshold:
                    action_reward = -2  # 体力过低，惩罚攻击
                else:
                    action_reward = 0  # 体力一般，不奖励也不惩罚
            # ... 其他动作奖励代码 ...

            # 组合动作奖励：如果上一个动作是向前翻滚或向前移动，当前动作是攻击，给予额外奖励
            if prev_action in [5, 9] and action in [1, 3]:
                combo_reward += 40  # 给予组合动作奖励

            # 连击逻辑
            if in_combo:
                if action in [1, 3] and boss_damage > 0:
                    # 连击中继续成功攻击
                    combo_count += 1
                    combo_reward += 100 * combo_count  # 连击奖励，奖励值可根据需要调整
                else:
                    # 连击中断
                    in_combo = False
                    combo_count = 0
            else:
                if action in [1, 3] and boss_damage > 0:
                    # 开始新的连击
                    in_combo = True
                    combo_count = 1
                    combo_reward += 10  # 初始连击奖励

            # 总奖励计算
            reward = self_blood_reward + boss_blood_reward + action_reward + stamina_penalty + idle_penalty + combo_reward

            done = 0
            emergence_break = 0
            return reward, done, stop, emergence_break, in_combo, combo_count



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
            time.sleep(3)
            directkeys.stop_forward()
        elif action == 6:  # 往后走s
            directkeys.run_backward()
            time.sleep(2)
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

    def suicide_restart(self):
        directkeys.menu()
        time.sleep(0.5)
        directkeys.switch_item()
        time.sleep(0.5)
        directkeys.action()
        time.sleep(3)
        directkeys.action()
        time.sleep(3)
        directkeys.switch_left_weapon()
        time.sleep(1)
        directkeys.action()
        time.sleep(5)
        print("Suicide Restart")

    def restart(self):
        i = 0
        while True:
            time.sleep(1)
            status, self_blood, self_stamina, boss_blood = self.get_status_info(
            )
            print("restart -> self_blood: ", self_blood, "self_stamina: ", self_stamina,
                  "boss_blood: ", boss_blood)
            i += 1
            if self_blood > 200 and self_stamina > 100:
                print('restart ')
                break
            if i >= 100:
                self.suicide_restart()
                i = 0
                continue
        time.sleep(3)
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
