# -*- coding: utf-8 -*-
import directkeys
import time


def restart():
    print("dead,restart")
    time.sleep(8)
    directkeys.lock_vision()
    time.sleep(0.2)
    directkeys.attack()
    print("restart a new episode")


if __name__ == "__main__":
    restart()
