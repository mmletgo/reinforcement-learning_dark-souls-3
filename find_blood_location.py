# -*- coding: utf-8 -*-
import cv2
import time
import grabscreen
import numpy as np

image_counter = 1

def calculate_health_percentage(color_image, label, red_threshold=150, green_blue_threshold=100):
    red_channel = color_image[4, :, 2] 
    green_channel = color_image[4, :, 1]
    blue_channel = color_image[4, :, 0] 

    self_blood = 0
    for pixel in color_image[4]:
        red_value = pixel[2]
        green_value = pixel[1]
        blue_value = pixel[0]

        if red_value > red_threshold and green_value < green_blue_threshold and blue_value < green_blue_threshold:
            self_blood += 1

    total_pixels = color_image.shape[1]
    health_percentage = (self_blood / total_pixels) * 100

    return health_percentage

def calculate_self_blood(color_image, label, red_self_blood_threshold=80,green_self_blood_threshold = 80,blue_self_blood_threshold =80 ):
    # red_channel = color_image[4, :, 2]
    # green_channel = color_image[4, :, 1]
    # blue_channel = color_image[4, :, 0]

    # # 打印红、绿、蓝通道的完整信息
    # print('label number:',label)
    # print(f"Label: {label} - Red channel matrix:\n", red_channel)
    # print(f"Label: {label} - Green channel matrix:\n", green_channel)
    #print(f"Label: {label} - Blue channel matrix:\n", blue_channel)
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

def calculate_boss_blood(color_image, label, red_boss_blood_threshold=70, self_stamina_green=30, self_stamina_blue=30):
    self_blood = 0
    for pixel in color_image[4]:
        red_value = pixel[2]
        green_value = pixel[1]
        blue_value = pixel[0]

        if red_value > red_boss_blood_threshold and green_value < self_stamina_green and blue_value < self_stamina_blue:
            self_blood += 1

    total_pixels = color_image.shape[1]
    health_percentage = (self_blood / total_pixels) * 100

    return health_percentage

def calculate_self_stamina(color_image, label, self_stamina_red=80, self_stamina_green=110, self_stamina_blue=80):
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

# 绘制黄色矩形框
def draw_bounding_box(image, top_left, bottom_right, color=(0, 255, 255), thickness=2):
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

# 打印文字到图片上
def print_text_on_image(image, text, position, color=(0, 255, 255), font_scale=1, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, thickness)

wait_time = 1
L_t = 3

window_size = (0, 0, 1920, 1080)  # 全屏
boss_blood_window = (557, 899, 1557, 910)  # Boss 血条区域
self_blood_window = (197, 75, 453, 89)  # 玩家血条区域
self_stamina_window = (197, 109, 419, 123)  # 玩家体力条区域

for i in list(range(wait_time))[::-1]:
    print(i + 1)
    time.sleep(1)

last_time = time.time()

while True:
    # 获取屏幕截图
    screen_image = grabscreen.grab_screen(window_size)
    
    self_screen_color = screen_image[self_blood_window[1]:self_blood_window[3], self_blood_window[0]:self_blood_window[2]]
    boss_screen_color = screen_image[boss_blood_window[1]:boss_blood_window[3], boss_blood_window[0]:boss_blood_window[2]]
    stamina_screen_color = screen_image[self_stamina_window[1]:self_stamina_window[3], self_stamina_window[0]:self_stamina_window[2]]

    # 计算血量和体力值
    self_blood = calculate_self_blood(self_screen_color, image_counter)
    boss_blood = calculate_boss_blood(boss_screen_color, image_counter)
    self_stamina = calculate_self_stamina(stamina_screen_color, image_counter)
    
    #debug--------------------------------------------------------------------------------
    # text = f"Player blood: {self_blood:.2f}% | Boss blood: {boss_blood:.2f}% | Stamina: {self_stamina:.2f}%"
    # print_text_on_image(screen_image, text, position=(599, 50), color=(0, 255, 255), font_scale=1, thickness=2)
    
    # draw_bounding_box(screen_image, (self_blood_window[0], self_blood_window[1]), (self_blood_window[2], self_blood_window[3]))
    # draw_bounding_box(screen_image, (boss_blood_window[0], boss_blood_window[1]), (boss_blood_window[2], boss_blood_window[3]))
    # draw_bounding_box(screen_image, (self_stamina_window[0], self_stamina_window[1]), (self_stamina_window[2], self_stamina_window[3]))

    # cv2.imwrite(f"processed_screen_{image_counter}.png", screen_image)
    #--------------------------------------------------------------------------------------

    # 显示图片（可选）
    # cv2.imshow('Processed Image', screen_image)

    # 记录处理时间
    last_time = time.time()

    # 增加计数器
    image_counter += 1

    # 按 'q' 键退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 视频结束后按任意键退出
cv2.waitKey()
cv2.destroyAllWindows()
