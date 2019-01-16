#이동

from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll
from pynput.keyboard import Key, Controller

user32 = windll.user32
user32.SetProcessDPIAware()
keyboard = Controller()

def FindPlayer(mask_pl):
    arr = numpy.where(mask_pl != 0)
    # print(arr)
    if len(arr[0]) != 0 or len(arr[1]) != 0:
        return arr[0][0], arr[1][0]
    return -1, -1

def MaskMap(im):
    lower_pl = numpy.array([15, 15, 175])
    upper_pl = numpy.array([35, 35, 190])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)

    return FindPlayer(mask_pl)

def MaskWall(im):
    lower_pl = numpy.array([0, 0, 0])
    upper_pl = numpy.array([25, 13, 19])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)
    arr = numpy.where(mask_pl)
    return mask_pl

def MakeROI(im, y, x):
    minX = x - 50 if x >= 50 else 0
    maxX = x + 50 if x <= 220 else 270
    minY = y - 50 if y >= 50 else 0
    maxY = y + 50 if y <= 350 else 0
    # print(minX, maxX, minY, maxY)
    # print(x, y)
    return im[minX:maxX, minY:maxY]

def Findaround(maskMap, x, y):
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        x_ = x
        y_ = y
        if(i%3 == 0):
            x_ -= 1
        elif(i%3 == 2):
            x_ += 1
        if(int(i/3) == 0):
            y_ -= 1
        elif(int(i/3) == 2):
            y_ += 1
        if maskMap[x_, y_] > 0:
            return 1
    return 0


def ccw(x1, y1, x2, y2, x3, y3):
    temp = x1 * y2 + x2 * y3 + x3 * y1
    if temp > 0:
        return 1
    elif temp < 0:
        return -1
    else:
        return 0

#경로 설정
moveList = [(220, 100)]

#경로 설정
isMoving = False

img = cv2.imread("map.png")
img2 = cv2.imread("map25.png")
img = cv2.add(img, img2)
cv2.imshow("q", img)
mask_ = MaskWall(img)

image, contours, hierachy = cv2.findContours(mask_, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
image = img

cnt = contours[6]

hull = cv2.convexHull(cnt)
for idx, arr in enumerate(hull):
    # print(hull[idx % len(hull)][0])
    x, y = hull[idx][0]
    x2, y2 = hull[(idx + 1) % len(hull)][0]
    cv2.line(img, (x, y), (x2, y2), (255, 0, 0), 1)
    print(x, y, x2, y2)
# cv2.line(img, (296, 164), (281, 174), (255, 0, 0), 1)
cv2.circle(img, (260, 160), 3, (255, 5, 55), -1)
cv2.circle(img, (230, 154), 3, (255, 5, 55), -1)
cv2.circle(img, (220, 170), 3, (255, 5, 55), -1)

print(ccw(260, 160, 230, 154, 220, 170))


# print(hull)
# defects = cv2.convexityDefects(cnt, hull)

# for i in range(defects.shape[0]):
#     sp, ep, fp, dist = defects[i, 0]
#     start = tuple(cnt[sp][0])
#     end = tuple(cnt[ep][0])
#     farthest = tuple(cnt[fp][0])
#     cv2.circle(img, farthest, 1, (255, 255, 255), -1)

cv2.imshow("w", mask_)
cv2.imshow("e", image)
cv2.waitKey(0)
cv2.destroyAllWindows()