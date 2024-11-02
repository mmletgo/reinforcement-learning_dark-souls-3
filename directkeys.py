# -*- coding: utf-8 -*-
"""
Modified version for game controls
"""

import ctypes
import time
import win32api
import win32con

SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
E = 0x12
R = 0x13
F = 0x21
Q = 0x10
ALT = 0x38
SPACE = 0x39
LSHIFT = 0x2A

UP = 0xC8
DOWN = 0xD0
LEFT = 0xCB
RIGHT = 0xCD

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def left_click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def right_click():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

def attack_right():
    left_click()

def heavy_attack_right():
    PressKey(LSHIFT)
    left_click()
    ReleaseKey(LSHIFT)

def attack_left():
    right_click()

def heavy_attack_left():
    PressKey(LSHIFT)
    right_click()
    ReleaseKey(LSHIFT)

def use_item():
    PressKey(R)
    time.sleep(0.1)
    ReleaseKey(R)

def action():
    PressKey(E)
    time.sleep(0.1)
    ReleaseKey(E)

def two_hand_weapon():
    PressKey(F)
    time.sleep(0.1)
    ReleaseKey(F)

def toggle_walk_run():
    PressKey(ALT)
    time.sleep(0.1)
    ReleaseKey(ALT)

def run_forward():
    PressKey(W)
    time.sleep(0.1)

def stop_forward():
    ReleaseKey(W)

def run_backward():
    PressKey(S)
    time.sleep(0.1)

def stop_backward():
    ReleaseKey(S)

def run_left():
    PressKey(A)
    time.sleep(0.1)

def stop_left():
    ReleaseKey(A)

def run_right():
    PressKey(D)
    time.sleep(0.1)

def stop_right():
    ReleaseKey(D)

def sprint_jump_roll():
    PressKey(SPACE)
    time.sleep(0.1)
    ReleaseKey(SPACE)

def reset_camera():
    PressKey(Q)
    time.sleep(0.1)
    ReleaseKey(Q)

def switch_spell():
    PressKey(UP)
    time.sleep(0.1)
    ReleaseKey(UP)

def switch_item():
    PressKey(DOWN)
    time.sleep(0.1)
    ReleaseKey(DOWN)

def switch_left_weapon():
    PressKey(LEFT)
    time.sleep(0.1)
    ReleaseKey(LEFT)

def switch_right_weapon():
    PressKey(RIGHT)
    time.sleep(0.1)
    ReleaseKey(RIGHT)


# if __name__ == '__main__':
#     time.sleep(5)  # 等待5秒
#     time1 = time.time()
#     while(True):
#         if abs(time.time()-time1) > 5:
#             break
#         else:
#             PressKey(W)
#             time.sleep(0.1)
#             ReleaseKey(W)
#             time.sleep(0.2)



def test_controls():
    print("控制测试程序启动...")
    print("请确保您在5秒内切换到目标窗口")
    time.sleep(5)
    
    def test_action(action_func, action_name):
        print(f"测试 {action_name}...")
        action_func()
        time.sleep(0.5)  
        
    # 测试方案
    test_sequences = [
        # 基础攻击测试
        (attack_right, "右手普通攻击"),
        (heavy_attack_right, "右手重击"),
        (attack_left, "左手普通攻击"),
        (heavy_attack_left, "左手重击"),
        
        # 移动测试
        (lambda: (run_forward(), time.sleep(1), stop_forward()), "向前移动1秒"),
        (lambda: (run_backward(), time.sleep(1), stop_backward()), "向后移动1秒"),
        (lambda: (run_left(), time.sleep(1), stop_left()), "向左移动1秒"),
        (lambda: (run_right(), time.sleep(1), stop_right()), "向右移动1秒"),
        
        # 功能键测试
        (use_item, "使用物品"),
        (action, "互动"),
        (two_hand_weapon, "双手持武器"),
        (sprint_jump_roll, "翻滚"),
        (reset_camera, "重置镜头"),
        
        # 切换测试
        (switch_spell, "切换法术"),
        (switch_item, "切换物品"),
        (switch_left_weapon, "切换左手武器"),
        (switch_right_weapon, "切换右手武器")
    ]
    
    try:
        for action_func, action_name in test_sequences:
            test_action(action_func, action_name)
            # response = input(f"{action_name} 测试完成。按Enter继续下一个测试，输入'q'退出: ")
            # if response.lower() == 'q':
            #     break
                
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        print("测试程序结束")

if __name__ == '__main__':
    test_controls()
