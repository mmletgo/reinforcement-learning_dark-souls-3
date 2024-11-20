# -*- coding: utf-8 -*-
import directkeys
import time

def restart():
    print("dead,restart")
    time.sleep(26)
    # time.sleep(2)
    directkeys.teleport()
    directkeys.teleport()

    time.sleep(1)
    directkeys.lock_target()
    time.sleep(1)
    directkeys.go_back()
    time.sleep(1)
    directkeys.action()
    time.sleep(4)
    directkeys.go_left()
    time.sleep(0.5)

    directkeys.go_forward_long()
    time.sleep(0.1)
    # directkeys.go_back()
    # time.sleep(0.5)
    directkeys.lock_target()

    print("restart a new episode")


if __name__ == "__main__":
    restart()
