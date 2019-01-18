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

#점들이 시계방향으로 정렬되어있는지, 반시계방향으로 정렬되어있는지, 일자로 늘어서 있는지 알아낸다
def ccw(x1, y1, x2, y2, x3, y3):
    temp = x1 * y2 + x2 * y3 + x3 * y1
    temp = temp - y1 * x2 - y2 * x3 - y3 * x1
    if temp > 0:
        return 1
    elif temp < 0:
        return -1
    else:
        return 0

#선분이 겹치는지 판정
def isIntersect(x, y, x2, y2, x3, y3, x4, y4):
    ab = ccw(x, y, x2, y2, x3, y3) * ccw(x, y, x2, y2, x4, y4)
    cd = ccw(x3, y3, x4, y4, x, y) * ccw(x3, y3, x4, y4, x2, y2)
    # print(ab, cd)

    #True 또는 False 반환
    return ab <= 0 and cd <= 0

def FindSlope(x, y, x2, y2):

    #y = ax * b 꼴일 때 a, b를 구하는 과정
    a = (y2 - y) / (x2 - x)
    b = y - a * x
    return a, b

#점의 위치를 바꾼다.
def ResizeLine(a, b, x, y, resize):
    y = a * (x + resize) + b
    x = (y - b) / a
    return x, y

#Contour를 감싸는 사각형을 더 크게 만든다. x, y와 x3, y3이 반대쪽 점이며, x2, y2와 x4, y4가 반대쪽 점이다.
def ResizeRect(x, y, x2, y2, x3, y3, x4, y4):
    if x > x3:
        x, y, x3, y3 = x3, y3, x, y
    if x2 > x4:
        x2, y2, x4, y4 = x4, y4, x2, y2
    
    print(x, y, x2, y2, x3, y3, x4, y4)
    a, b = FindSlope(x, y, x3, y3)
    a2, b2 = FindSlope(x2, y2, x4, y4)
    print(a, b, a2, b2)
    x, y = ResizeLine(a, b, x, y, -3)
    x2, y2 = ResizeLine(a2, b2, x2, y2, -3)
    x3, y3 = ResizeLine(a, b, x3, y3, 3)
    x4, y4 = ResizeLine(a2, b2, x4, y4, 3)
    return int(round(x)), int(round(y)), int(round(x2)), int(round(y2)), int(round(x3)), int(round(y3)), int(round(x4)), int(round(y4))

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
cv2.circle(img, (220, 170), 3, (255, 5, 55), -1)
cv2.circle(img, (260, 160), 3, (255, 5, 55), -1)

hull = cv2.convexHull(cnt)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = numpy.int0(box)
# cv2.drawContours(img,[box],0,(0,0,255),2)
x, y, x2, y2, x3, y3, x4, y4 = ResizeRect(box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1])
print(x, y, x2, y2, x3, y3, x4, y4)

cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
cv2.circle(img, (x2, y2), 1, (0, 255, 255), -1)
cv2.circle(img, (x3, y3), 1, (0, 255, 255), -1)
cv2.circle(img, (x4, y4), 1, (0, 255, 255), -1)

for idx, arr in enumerate(box):
    # print(hull[idx % len(hull)][0])
    x, y = box[idx]
    x2, y2 = box[(idx + 1) % len(box)]
    # cv2.line(img, (x, y), (x2, y2), (255, 0, 0), 1)
    # cv2.circle(img, (x, y), 3, (255, 5, 55), -1)
    cv2.imshow("e", img)
    ccwtmp = ccw(220, 170, x, y, 260, 160)
    # print(ccwtmp)

# cv2.circle(img ,(x, y), 1, (255, 0, 0), -1)
# cv2.line(img, (hull[0]), hull[int(len(hull)/2)], (255, 0, 0), 1)
# cv2.line(img, (296, 164), (281, 174), (255, 0, 0), 1)
# cv2.circle(img, (260, 160), 3, (255, 5, 55), -1)
# cv2.circle(img, (230, 154), 3, (255, 5, 55), -1)
# cv2.circle(img, (100, 100), 3, (255, 5, 55), -1)
# cv2.waitKey(0)
# cv2.imshow("e", image)

# cv2.circle(img, (60, 60), 3, (255, 5, 55), -1)
# cv2.waitKey(0)
# cv2.imshow("e", image)

# cv2.circle(img, (70, 40), 3, (255, 5, 55), -1)
# cv2.waitKey(0)
# cv2.imshow("e", image)

# cv2.circle(img, (260, 160), 1, (255, 5, 55), -1)
# cv2.circle(img, (244, 167), 1, (255, 5, 55), -1)
# cv2.circle(img, (228, 156), 1, (255, 5, 55), -1)
# cv2.line(img, (244, 167), (228, 156), (255, 0, 0), 1)
# cv2.line(img, (220, 170), (260, 160), (255, 0, 0), 1)

# print(isIntersect(220, 170, 260, 160, 244, 167, 228, 156))

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


# for idx, arr in enumerate(hull):
#     # print(hull[idx % len(hull)][0])
#     x, y = hull[idx][0]
#     x2, y2 = hull[(idx + 1) % len(hull)][0]
#     # cv2.line(img, (x, y), (x2, y2), (255, 0, 0), 1)
#     # cv2.circle(img, (x, y), 3, (255, 5, 55), -1)
#     cv2.imshow("e", img)
#     ccwtmp = ccw(220, 170, x, y, 260, 160)
#     if ccwtmp >= 1:
#         cv2.circle(img, (x, y - 5), 1, (0, 5, 55), -1)
#     # cv2.waitKey(0)
#     # print(isIntersect(220, 170, 260, 160, x, y, x2, y2))

#     # cv2.waitKey(0)
#     # cv2.imshow("e", image)
#     # print(x, y, x2, y2)