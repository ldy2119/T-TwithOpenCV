from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll

def FindPlayer(mask_pl):
    arr = numpy.where(mask_pl != 0)
    print(arr)
    if len(arr[0]) != 0 or len(arr[1]) != 0:
        return arr[0][0], arr[1][0]
    return -1, -1

def MaskMap(im):
    lower_pl = numpy.array([15, 15, 175])
    upper_pl = numpy.array([35, 35, 190])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)
    # adf = cv2.bitwise_and(im, im, mask = mask_pl)
    # cv2.imshow("e", adf)

    return FindPlayer(mask_pl)

def MaskWall(im):
    lower_pl = numpy.array([3, 0, 2])
    upper_pl = numpy.array([20, 13, 18])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)
    return mask_pl

img = cv2.imread("map.png")
img2 = cv2.imread("map25.png")

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_fg = cv2.bitwise_and(img, img, mask=mask_inv)
mask_ = MaskWall(img1_fg)
# img2_bg = cv2.bitwise_and(img, img, mask=mask_inv)

y, x = MaskMap(img1_fg)
# print(x, y)

point_nw = [(x + 5, y - 9), (x, y + 6), (x - 5, y), (x - 13, y + 3)]
point_ne = [(x - 5, y - 9), (x, y + 6), (x + 5, y), (x + 13, y + 3)]
point_sw = [(x + 5, y + 9), (x, y - 6), (x - 5, y), (x - 13, y - 3)]
point_se = [(x - 5, y + 9), (x, y - 6), (x + 5, y), (x + 13, y - 3)]
point = [point_nw, point_ne, point_sw, point_se]
index = 0
max = -1
maxindex = -1
for idx, arr in enumerate(point):
    tmp = 0
    for y, x in arr:
        if mask_[x, y] > 0:
            cv2.circle(img1_fg, (y, x), 1, (255, 0, 255), -1)
            tmp += 1
    if tmp > max:
            max = tmp
            maxindex = idx
    cv2.line(img1_fg, arr[0], arr[3], (255, 255, 255), 1)

print(max, maxindex)
cv2.imshow("q", img1_fg)
# print(mask_[y+9, x+5])

cv2.imshow("w", mask_)
# cv2.imshow("w", img2_bg)
# cv2.imshow("e", mask)

# dst = cv2.add(img1_fg, img2_bg)

cv2.waitKey(0)
cv2.destroyAllWindows()


