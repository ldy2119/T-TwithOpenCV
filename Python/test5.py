#맵 사각형으로 만들기

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

img = cv2.imread("map2.png")

rows, cols, ch = img.shape

ptr1 = numpy.float32([[200, 0], [400, 135], [200, 270]])
ptr2 = numpy.float32([[400, 0], [400, 270], [0, 270]])

M = cv2.getAffineTransform(ptr1, ptr2)

dst = cv2.warpAffine(img, M, (cols,rows))


lower_pl = numpy.array([3, 0, 2])
upper_pl = numpy.array([20, 13, 18])
mask_pl = cv2.inRange(dst, lower_pl, upper_pl)

cv2.imshow("ee", dst)
cv2.imshow("ww", mask_pl)

lower_pl = numpy.array([15, 15, 175])
upper_pl = numpy.array([35, 35, 190])
mask_pl2 = cv2.inRange(dst, lower_pl, upper_pl)

x, y = FindPlayer(mask_pl2)

minX = x - 50 if x >= 50 else 0
maxX = x + 50 if x <= 220 else 270
minY = y - 50 if y >= 50 else 0
maxY = y + 50 if y <= 350 else 0

print(minX, maxX, minY, maxY)

img2 = mask_pl[minX:maxX, minY:maxY]
dst = dst[minX : maxX, minY:maxY]

_, contours, _ = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(dst, contours, -1, (0, 255, 0), 3)
print(contours)
cv2.imshow("qq", dst)
cv2.waitKey(0)