from getkeys import key_check
import time
import directkeys
import cv2
from grabscreen import grab_screen
from setting import WIDTH, HEIGHT, window_size, self_blood_window, boss_blood_window, self_stamina_window


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


def get_status_info():
    screen_image = grab_screen(window_size)

    self_screen_color = cv2.cvtColor(
        screen_image[self_blood_window[1]:self_blood_window[3],
                     self_blood_window[0]:self_blood_window[2]],
        cv2.COLOR_BGR2GRAY)
    boss_screen_color = screen_image[boss_blood_window[1]:boss_blood_window[3],
                                     boss_blood_window[0]:boss_blood_window[2]]
    stamina_screen_color = cv2.cvtColor(
        screen_image[self_stamina_window[1]:self_stamina_window[3],
                     self_stamina_window[0]:self_stamina_window[2]],
        cv2.COLOR_BGR2GRAY)
    # 计算血量和体力值
    self_blood = self_blood_count_grey(self_screen_color)
    boss_blood = boss_blood_count(boss_screen_color)
    self_stamina = self_stamina_count_grey(stamina_screen_color)
    screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
    status = cv2.resize(screen_gray, (WIDTH, HEIGHT))
    return status, self_blood, self_stamina, boss_blood


###################################################################################################################
def self_blood_count_grey(grey_image):
    self_blood = 0
    for pixel in grey_image[4]:
        if pixel >= 90:
            self_blood += 1
    return self_blood


def self_stamina_count_grey(grey_image):
    self_blood = 0
    for pixel in grey_image[4]:
        if pixel >= 117:
            self_blood += 1
    return self_blood


def self_blood_count(color_image,
                     red_self_blood_threshold=80,
                     green_self_blood_threshold=80,
                     blue_self_blood_threshold=80):
    self_blood = 0
    for pixel in color_image[4]:
        red_value = pixel[2]
        green_value = pixel[1]
        blue_value = pixel[0]
        if red_value > red_self_blood_threshold and green_value < green_self_blood_threshold and blue_value < green_self_blood_threshold:
            self_blood += 1

    total_pixels = color_image.shape[1]
    health_percentage = (self_blood / total_pixels) * 100

    return health_percentage


def boss_blood_count(color_image,
                     red_boss_blood_threshold=50,
                     self_stamina_green=30,
                     self_stamina_blue=30):
    self_blood = 0
    for pixel in color_image[7]:
        red_value = pixel[2]
        green_value = pixel[1]
        blue_value = pixel[0]

        if red_value > red_boss_blood_threshold and green_value < self_stamina_green and blue_value < self_stamina_blue:
            self_blood += 1
    return self_blood


def self_stamina_count(color_image,
                       self_stamina_red=80,
                       self_stamina_green=110,
                       self_stamina_blue=80):
    self_blood = 0
    for pixel in color_image[4]:
        red_value = pixel[2]
        green_value = pixel[1]
        blue_value = pixel[0]

        if red_value > self_stamina_red and green_value > self_stamina_green and blue_value > self_stamina_blue:
            self_blood += 1

    total_pixels = color_image.shape[1]
    health_percentage = (self_blood / total_pixels) * 100

    return health_percentage


#############################################################################################################################
# TODO: need to be modified
def take_action(action):
    if action == 0:  # n_choose
        pass
    elif action == 1:  # 左击
        directkeys.left_click()
    elif action == 2:  # 右击（盾）
        directkeys.right_click()
    elif action == 3:  # 重击
        directkeys.heavy_attack_left()
    elif action == 4:  # r喝药
        directkeys.use_item()
    elif action == 5:  # 向后闪避，没加翻滚
        directkeys.sprint_jump_roll()
    elif action == 6:  # 往前走w
        directkeys.run_forward()
        time.sleep(1)
        directkeys.stop_forward()
    elif action == 7:  # 往后走s
        directkeys.run_backward()
        time.sleep(1)
        directkeys.stop_backward()
    elif action == 8:  # 往左走a
        directkeys.run_left()
        time.sleep(1)
        directkeys.stop_left()
    elif action == 9:  # 往右走d
        directkeys.run_right()
        time.sleep(1)
        directkeys.stop_right()
