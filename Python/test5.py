from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll

img = cv2.imread("map2.png")

rows, cols, ch = img.shape

ptr1 = numpy.float32([[200, 0], [400, 135], [200, 270]])
ptr2 = numpy.float32([[400, 0], [400, 270], [0, 270]])

M = cv2.getAffineTransform(ptr1, ptr2)

dst = cv2.warpAffine(img, M, (cols,rows))

cv2.imshow("ee", dst)
cv2.waitKey(0)