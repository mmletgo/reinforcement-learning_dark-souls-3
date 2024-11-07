# -*- coding: utf-8 -*-
import numpy as np
import win32gui, win32ui, win32con, win32api
import psutil
import win32process


def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


def get_hwnd_by_process_name(process_name="DarkSoulsIII.exe"):
    """根据进程名称获取窗口句柄"""
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'] == process_name:
            pid = proc.info['pid']
            # 遍历所有窗口，找到属于该进程的窗口句柄
            def callback(hwnd, hwnds):
                if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                    if found_pid == pid:
                        hwnds.append(hwnd)
                return True
            hwnds = []
            win32gui.EnumWindows(callback, hwnds)
            return hwnds[0] if hwnds else None
    return None


def grab_screen_by_process(hwnd):

    # 获取窗口位置和尺寸
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    # 获取标题栏高度和边框冗余
    titlebar_height = win32api.GetSystemMetrics(win32con.SM_CYCAPTION)
    print(titlebar_height)
    border_thickness = 8  # 假设每边都有8像素冗余

    # 计算实际内容区域
    # content_left = left + border_thickness
    # content_top = top + titlebar_height + border_thickness
    content_width = width - 2 * border_thickness
    content_height = height - titlebar_height - 2 * border_thickness

    # 创建与内容区域大小一致的位图
    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, content_width, content_height)
    memdc.SelectObject(bmp)

    # 从去掉冗余的区域开始截图
    memdc.BitBlt((0, 0), (content_width, content_height), srcdc, (border_thickness, titlebar_height + border_thickness), win32con.SRCCOPY)

    # 获取位图数据并转换为numpy数组
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (content_height, content_width, 4)  # BGRA格式

    # 释放资源
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

# # 示例：指定进程名进行截图
# process_name = "notepad.exe"  # 替换为你的目标进程名称
# img = grab_screen_by_process()
# print("截图已完成，返回的图像数组尺寸为：", img.shape)
