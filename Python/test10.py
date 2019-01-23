#마우스 클릭 구현


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

#28 117 198
#30 126 214
#RGB로, 상대편을 나타냄