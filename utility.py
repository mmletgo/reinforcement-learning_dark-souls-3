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


# calculate self blood
# TODO: need to be modified
def self_blood_count(self_gray):
    self_blood = 0
    for self_bd_num in self_gray[469]:
        # self blood gray pixel 80~98
        if self_bd_num > 90 and self_bd_num < 98:
            self_blood += 1
    return self_blood


# calculate boss blood
# TODO: need to be modified
def boss_blood_count(boss_gray):
    boss_blood = 0
    for boss_bd_num in boss_gray[0]:
        # boss blood gray pixel 65~75
        if boss_bd_num > 65 and boss_bd_num < 75:
            boss_blood += 1
    return boss_blood


# TODO: need to be modified
def take_action(action):
    if action == 0:  # n_choose
        pass
    elif action == 1:  # j
        directkeys.attack()
    elif action == 2:  # k
        directkeys.jump()
    elif action == 3:  # m
        directkeys.defense()
    elif action == 4:  # r
        directkeys.dodge()
