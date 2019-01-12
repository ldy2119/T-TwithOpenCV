from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from pynput.mouse import Button, Controller as MouseCtrl

mouse = MouseCtrl()

mouse.press(Button.left)
mouse.release(Button.left)
mouse.press(Button.left)
mouse.release(Button.left)