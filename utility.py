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
        self.cooldown_counter = 0

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

        #Draw rectangles and save annotated image
        annotated_image = screen_image_rgb.copy()
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
        #print(f"Status shape: {status.shape}")
        #print('status',status[100])

        return status, self.self_blood, self_stamina, self.boss_blood

    def action_judge(self, self_blood, next_self_blood, self_stamina, next_self_stamina,
                 boss_blood, next_boss_blood, action, prev_action, stop, emergence_break,
                 in_combo, combo_count):
        high_stamina_threshold = 50
        low_stamina_threshold = 20

        if self.cooldown_counter > 0:
            if action in [1, 3,5,9]:
                reward = -200
                done = 0
                stop = 0
                emergence_break += 1 if emergence_break < 2 else 100
                return reward, done, stop, emergence_break, in_combo, combo_count
            else:
                self.cooldown_counter -= 1
                reward = 20 
                done = 0
                stop = 0
                return reward, done, stop, emergence_break, in_combo, combo_count

        if next_self_blood < 3:
            reward = -2000
            done = 1
            stop = 0
            emergence_break += 1 if emergence_break < 2 else 100
            return reward, done, stop, emergence_break, False, 0
        elif next_boss_blood < 3:
            reward = 3000
            done = 0
            stop = 0
            emergence_break += 1 if emergence_break < 2 else 100
            return reward, done, stop, emergence_break, False, 0
        else:
            self_blood_reward = 0
            boss_blood_reward = 0
            action_reward = 0
            stamina_penalty = 0
            idle_penalty = 0
            combo_reward = 0

            if next_self_blood < self_blood:
                if self_blood>180:
                    self_blood_reward = (next_self_blood - self_blood) * 3
                else:
                    self_blood_reward = (next_self_blood - self_blood) * 4
                    
            if next_self_blood - self_blood > -1:
                self_blood_reward = 20

            boss_damage = boss_blood - next_boss_blood
            if boss_damage > 0:
                boss_blood_reward = boss_damage * 3

            current_stamina = next_self_stamina
            
            if in_combo:
                if action in [1, 3] and boss_damage > 0:
                    combo_count += 1
                    combo_reward += 40 * combo_count
                    if combo_count >= 3:
                        self.cooldown_counter = 1
                        in_combo = False
                        combo_count = 0
                        print("cool down！")
                else:
                    in_combo = False
                    combo_count = 0

            if action in [1, 3,9,10,11,12]:
                if current_stamina > high_stamina_threshold:
                    action_reward = 10
                elif current_stamina < low_stamina_threshold:
                    action_reward = -10
                else:
                    action_reward = 0

            if prev_action in [5,7,11,12, 9] and action in [1, 3]:
                combo_reward += 40
            
            if action == 2:
                action_reward = 20
            else:
                if action in [1, 3] and boss_damage > 0:
                    in_combo = True
                    combo_count = 1
                    combo_reward += 10
                    
            if action == 0:
                action_reward += -50
                print("do nothing, punish！")

            reward = self_blood_reward + boss_blood_reward + action_reward + stamina_penalty + idle_penalty + combo_reward

            done = 0
            emergence_break = 0
            return reward, done, stop, emergence_break, in_combo, combo_count




    def take_action(self, action):
        if action == 0: 
            pass
        elif action == 1: 
            directkeys.left_click()
        elif action == 2: 
            directkeys.right_click()
        elif action == 3: 
            directkeys.heavy_attack_left()
        elif action == 4: 
            directkeys.sprint_jump_roll()
        elif action == 5: 
            directkeys.run_forward()
            time.sleep(2)
            directkeys.stop_forward()
        elif action == 6: 
            directkeys.run_backward()
            time.sleep(2)
            directkeys.stop_backward()
        elif action == 7:
            directkeys.run_left()
            time.sleep(0.1)
            directkeys.stop_left()
        elif action == 8: 
            directkeys.run_right()
            time.sleep(0.1)
            directkeys.stop_right()
        elif action == 9:
            directkeys.run_forward()
            directkeys.sprint_jump_roll()
            time.sleep(0.1)
            directkeys.stop_forward()
        elif action == 10:
            directkeys.run_backward()
            directkeys.sprint_jump_roll()
            time.sleep(0.1)
            directkeys.stop_backward()
        elif action == 11:
            directkeys.run_left()
            directkeys.sprint_jump_roll()
            time.sleep(0.1)
            directkeys.stop_left()
        elif action == 12:
            directkeys.run_right()
            directkeys.sprint_jump_roll()
            time.sleep(0.1)
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
            if i >= 70:
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
        time.sleep(5)
        directkeys.stop_forward()
        time.sleep(0.2)
        
        directkeys.reset_camera()
        print("restart a new episode")
