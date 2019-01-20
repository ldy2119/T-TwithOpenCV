#배럭 짓기

from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll
from pynput.keyboard import Key, Controller

user32 = windll.user32
user32.SetProcessDPIAware()
keyboard = Controller()

def BuildBarrack():
    keyboard.press(Key.space)