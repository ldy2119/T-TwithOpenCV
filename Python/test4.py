# import cv2

# from pynput.keyboard import Key, Controller
# import time

# keyboard = Controller()

# keyboard.press('a')
# time.sleep(1)
# keyboard.release('a')

from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll

user32 = windll.user32
user32.SetProcessDPIAware()


image = ImageGrab.grab(bbox=(0, 800, 400, 1070))
printScreen = numpy.array(image)
printScreen = cv2.imread("map25.png")

img2 = cv2.imread("map.png")
dst = cv2.add(img2, printScreen)

lower_pl = numpy.array([15, 15, 175])
upper_pl = numpy.array([35, 35, 190])
mask_pl = cv2.inRange(dst, lower_pl, upper_pl)
# for a in mask_pl:
#     for b in a:
#         if b != 0:
#             print(b)
#             print([numpy.where(a == 255)])
for a in [numpy.where(mask_pl != 0)]:
    print(a[0][0], a[1][0])
# printScreen = cv2.imread("map23.png")
im = mask_pl[167:170, 270:275]

# print(mask_pl[270, 273])
cv2.imshow("a", mask_pl)
cv2.waitKey(0)

# 20 20 187

# 35 35 200