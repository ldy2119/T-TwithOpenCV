#농장 찾기(8과 합쳐야 함, 수정 바람)

from PIL import ImageGrab
from PIL import Image
import numpy
import cv2


def MaskMap(im, mask):
    dst = cv2.add(im, mask)
    return dst

mask_ = [cv2.imread("map61.png"), cv2.imread("map62.png"), cv2.imread("map63.png"), cv2.imread("map64.png"),
         cv2.imread("map65.png"), cv2.imread("map66.png"), cv2.imread("map67.png"), cv2.imread("map68.png"),
         cv2.imread("map69.png")]

img = cv2.imread("map.png")

i = 0
for imgg in mask_:
    im = MaskMap(img, imgg)
    cv2.imshow(str(i), im)
    i+=1

cv2.waitKey(0)
cv2.destroyAllWindows()