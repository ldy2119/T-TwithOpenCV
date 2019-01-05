from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll

def FindPlayer(mask_pl):
    arr = numpy.where(mask_pl != 0)
    if len(arr[0]) != 0 or len(arr[1]) != 0:
        return arr[0][0], arr[1][0]
    return -1, -1

def MaskWall(img):
    lower_pl = numpy.array([3, 0, 2])
    upper_pl = numpy.array([20, 13, 18])
    mask_pl = cv2.inRange(img, lower_pl, upper_pl)
    cv2.imshow("ma", mask_pl)
    # return FindPlayer(mask_pl)

def MaskMap(img, mapMaskImage):
    dst = cv2.add(img, mapMaskImage)
    return MaskWall(dst)

img = cv2.imread("map2.png")

mapMaskImage = cv2.imread("map25.png")

# x, y = MaskMap(img, mapMaskImage)
MaskMap(img, mapMaskImage)

cv2.waitKey(0)

cv2.destroyAllWindows()

# print(x, y)