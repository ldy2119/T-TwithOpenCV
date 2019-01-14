#농장 찾기(8과 합쳐야 함, 수정 바람)

from PIL import ImageGrab
from PIL import Image
import numpy
import cv2


def MaskMap(im, mask):
    dst = cv2.add(im, mask)
    return dst

mask_ = [cv2.imread("map6-1.png"), cv2.imread("map6-2.png"), cv2.imread("map6-3.png"), cv2.imread("map6-4.png"),
         cv2.imread("map6-5.png"), cv2.imread("map6-6.png"), cv2.imread("map6-7.png"), cv2.imread("map6-8.png"),
         cv2.imread("map6-9.png")]

img = cv2.imread("map.png")

i = 0
for imgg in mask_:
    im = MaskMap(img, imgg)
    cv2.imshow(str(i), im)
    i+=1

cv2.waitKey(0)
cv2.destroyAllWindows()