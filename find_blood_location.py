# -*- coding: utf-8 -*-
import cv2
import time
import grabscreen


def self_blood_count(self_gray):
    self_blood = 0
    for self_bd_num in self_gray[469]:
        # self blood gray pixel 80~98
        # 血量灰度值80~98
        print(self_bd_num)
        if self_bd_num > 90 and self_bd_num < 98:
            self_blood += 1
    return self_blood


def boss_blood_count(boss_gray):
    boss_blood = 0
    for boss_bd_num in boss_gray[0]:
        # boss blood gray pixel 65~75
        # 血量灰度值65~75
        # print(boss_bd_num)
        if boss_bd_num > 65 and boss_bd_num < 75:
            boss_blood += 1
    return boss_blood


wait_time = 1
L_t = 3

window_size = (20, 40, 1900, 1060)
self_blood_window = (60, 91, 280, 562)
boss_blood_window = (60, 91, 280, 562)

for i in list(range(wait_time))[::-1]:
    print(i + 1)
    time.sleep(1)

last_time = time.time()
while (True):
    screen_gray = cv2.cvtColor(grabscreen.grab_screen(window_size),
                               cv2.COLOR_BGR2GRAY)  # 灰度图像收集

    # screen_gray = cv2.cvtColor(grabscreen.grab_screen(self_blood_window),
    #                            cv2.COLOR_BGR2GRAY)  # 灰度图像收集
    # self_blood = self_blood_count(screen_gray)

    # screen_gray = cv2.cvtColor(grabscreen.grab_screen(boss_blood_window),
    #                            cv2.COLOR_BGR2GRAY)  # 灰度图像收集
    # boss_blood = boss_blood_count(screen_gray)

    cv2.imshow('window1', screen_gray)

    # 测试时间用
    print('loop took {} seconds'.format(time.time() - last_time))
    last_time = time.time()

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()  # 视频结束后，按任意键退出
cv2.destroyAllWindows()
