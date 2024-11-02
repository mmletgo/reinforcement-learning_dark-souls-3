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
    for pixel in color_image[4]:
        red_value = pixel[2]
        green_value = pixel[1]
        blue_value = pixel[0]
        if red_value > red_self_blood_threshold and green_value < green_self_blood_threshold and blue_value < green_self_blood_threshold:
            self_blood += 1

    total_pixels = color_image.shape[1]
    health_percentage = (self_blood / total_pixels) * 100

    return health_percentage

def boss_blood_count(color_image, red_boss_blood_threshold=70, self_stamina_green=30, self_stamina_blue=30):
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

def self_stamina_count(color_image, self_stamina_red=80, self_stamina_green=110, self_stamina_blue=80):
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

'''
def self_blood_count(color_image, red_self_blood_threshold=80, green_self_blood_threshold=80, blue_self_blood_threshold=80):
    try:
        if len(color_image.shape) != 3:
            print(f"Warning: Image should be 3 channels, but got shape {color_image.shape}")
            return 0
            
        self_blood = 0
        height = color_image.shape[0]
        width = color_image.shape[1]
        
        row = 4  
        if row >= height:
            print(f"Warning: Row 4 is out of bounds. Image height is {height}")
            return 0
            
        for x in range(width):
            pixel = color_image[row, x]  
            blue_value = pixel[0]
            green_value = pixel[1]
            red_value = pixel[2]
            
            if (red_value > red_self_blood_threshold and 
                green_value < green_self_blood_threshold and 
                blue_value < blue_self_blood_threshold):
                self_blood += 1

        health_percentage = (self_blood / width) * 100
        return health_percentage
        
    except Exception as e:
        print(f"Error in self_blood_count: {e}")
        print(f"Image shape: {color_image.shape}")
        return 0

def boss_blood_count(color_image, red_boss_blood_threshold=70, self_stamina_green=30, self_stamina_blue=30):
    try:
        if len(color_image.shape) != 3:
            print(f"Warning: Image should be 3 channels, but got shape {color_image.shape}")
            return 0
            
        self_blood = 0
        height = color_image.shape[0]
        width = color_image.shape[1]
        
        row = 4
        if row >= height:
            print(f"Warning: Row 4 is out of bounds. Image height is {height}")
            return 0
            
        for x in range(width):
            pixel = color_image[row, x]
            blue_value = pixel[0]
            green_value = pixel[1]
            red_value = pixel[2]

            if (red_value > red_boss_blood_threshold and 
                green_value < self_stamina_green and 
                blue_value < self_stamina_blue):
                self_blood += 1

        health_percentage = (self_blood / width) * 100
        return health_percentage
        
    except Exception as e:
        print(f"Error in boss_blood_count: {e}")
        print(f"Image shape: {color_image.shape}")
        return 0

def self_stamina_count(color_image, self_stamina_red=80, self_stamina_green=110, self_stamina_blue=80):
    try:
        if len(color_image.shape) != 3:
            print(f"Warning: Image should be 3 channels, but got shape {color_image.shape}")
            return 0
            
        self_blood = 0
        height = color_image.shape[0]
        width = color_image.shape[1]
        
        row = 4
        if row >= height:
            print(f"Warning: Row 4 is out of bounds. Image height is {height}")
            return 0
            
        for x in range(width):
            pixel = color_image[row, x]
            blue_value = pixel[0]
            green_value = pixel[1]
            red_value = pixel[2]

            if (red_value > self_stamina_red and 
                green_value > self_stamina_green and 
                blue_value > self_stamina_blue):
                self_blood += 1

        health_percentage = (self_blood / width) * 100
        return health_percentage
        
    except Exception as e:
        print(f"Error in self_stamina_count: {e}")
        print(f"Image shape: {color_image.shape}")
        return 0
'''

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
