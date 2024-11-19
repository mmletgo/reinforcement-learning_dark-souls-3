from getkeys import key_check
import time
import directkeys


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



###################################################################################################################
def self_blood_count(color_image, red_self_blood_threshold=80,green_self_blood_threshold = 80,blue_self_blood_threshold =80 ):
    self_blood = 0
    for pixel in color_image[-1]:
        
        red_value = pixel[2]
        green_value = pixel[1]
        blue_value = pixel[0]
        if red_value > red_self_blood_threshold and green_value < green_self_blood_threshold and blue_value < green_self_blood_threshold:
            self_blood += 1

    total_pixels = color_image.shape[1]
    health_percentage = (self_blood / total_pixels) * 100

    return health_percentage

def boss_blood_count(color_image, red_boss_blood_threshold=40, self_stamina_green=30, self_stamina_blue=30):
    self_blood = 0
    for pixel in color_image[-1]:
        red_value = pixel[2]
        green_value = pixel[1]
        blue_value = pixel[0]

        if red_value > red_boss_blood_threshold and green_value < self_stamina_green and blue_value < self_stamina_blue:
            self_blood += 1

    total_pixels = color_image.shape[1]
    health_percentage = (self_blood / total_pixels) * 100

    return health_percentage

def self_stamina_count(color_image, self_stamina_red=80, self_stamina_green=110, self_stamina_blue=80):
    self_blood = 0
    for pixel in color_image[-1]:
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
#action_space = {1: 'forward', 2: 'backward', 3: 'left', 4: 'right', 5: 'shield', 6: 'sword_light', 7: 'sword_heavy',8: 'shield_heavy', 9: 'jump_forward', 10: 'jump_left', 11: 'jump_right', 12: 'jump', 13: 'use_item'}
def take_action(action):
    if action == 0:  # n_choose
        pass
    elif action == 1:  # j
        directkeys.go_forward()
    elif action == 2:  # k
        directkeys.go_back()
    elif action == 3:  # m
        directkeys.go_left()
    elif action == 4:  # r
        directkeys.go_right()
    elif action == 5:  # j
        directkeys.shield()
    elif action == 6:  # k
        directkeys.sword_light()
    elif action == 7:  # m
        directkeys.sword_heavy()
    elif action == 8:  # r
        directkeys.shield_heavy()
    elif action == 9:  # j
        directkeys.jump_forward()
    elif action == 10:  # k
        directkeys.jump_left()
    elif action == 11:  # m
        directkeys.jump_right()
    elif action == 12:  # r
        directkeys.jump_backward()
    elif action == 13:  # r
        directkeys.use_item()
    else:
        pass
