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
    # arr = numpy.where(mask_pl)
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
def IsIntersect(x, y, x2, y2, x3, y3, x4, y4):
    #x, y와 x2, y2가 선분.(현재 위치와 목적지) x3, y3와 x4, y4가 선분(겹치는지 확인할 벽)
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
    
    # print(x, y, x2, y2, x3, y3, x4, y4)
    a, b = FindSlope(x, y, x3, y3)
    a2, b2 = FindSlope(x2, y2, x4, y4)
    # print(a, b, a2, b2)
    x, y = ResizeLine(a, b, x, y, -3)
    x2, y2 = ResizeLine(a2, b2, x2, y2, -3)
    x3, y3 = ResizeLine(a, b, x3, y3, 3)
    x4, y4 = ResizeLine(a2, b2, x4, y4, 3)
    return int(round(x)), int(round(y)), int(round(x2)), int(round(y2)), int(round(x3)), int(round(y3)), int(round(x4)), int(round(y4))


def IsCrossRect(x, y, x2, y2, x3, y3, x4, y4, dirx, diry, curx, cury):
    arr = [(x, y), (x2, y2), (x3, y3), (x4, y4)]
    for idx, obj in enumerate(arr):
        if IsIntersect(curx, cury, dirx, diry, arr[idx][0], arr[idx][1], arr[(idx + 1) % 4][0], arr[(idx + 1) % 4][1]) == True:
            return True
    return False

def ccwRect(x, y, x2, y2, x3, y3, x4, y4, dirx, diry, curx, cury):
    moveList = []
    arr = [(x, y), (x2, y2), (x3, y3), (x4, y4)]
    ccwtemp = []
    for tmp in arr:
        ccwtemp.append(ccw(curx, cury, tmp[0], tmp[1], dirx, diry))

    for idx, tmp in enumerate(ccwtemp):
        if ccwtemp[idx] == -1:
            moveList.insert(0, (arr[idx][0], arr[idx][1]))
    return moveList



def ReleaseAll():
    keyboard.release('r')
    keyboard.release('a')
    keyboard.release('s')
    keyboard.release('d')
    keyboard.release('w')
    keyboard.release(Key.space)

#북쪽으로 이동
def MoveN():
    keyboard.press('w')

#동쪽으로 이동
def MoveE():
    keyboard.press('d')

#남쪽으로 이동
def MoveS():
    keyboard.press('s')

#서쪽으로 이동
def MoveW():
    keyboard.press('a')

def FindWay(im, x, y, x_, y_):
    x__, y__ = 0, 0
    if abs(x_ - x) < abs(y_ - y):
        y__ = y_
        x__ = x
    else:
        y__ = y
        x__ = x_
    # cv2.circle(im, (x__, y__), 1, (255, 0, 0), -1)
    # print(x__, y__, abs(x_ - x), abs(y_ - y))
    moveList.insert(0, (x__, y__))
    print(x_, y_)
    return True

def Move(x, y, x_, y_):
    value = -1

    ReleaseAll()
    if abs(x_ - x) > 5:
        if x < x_:
            MoveE()
        elif x > x_:
            MoveW()
        value = 1
    
    if abs(y_ - y) > 5:
        if y < y_:
            MoveS()
        elif y > y_:
            MoveW()
        value = 1
    
    return value

# #경로 설정
# moveList = [(160, 160)]

# #경로 설정
# isMoving = False

# img = cv2.imread("map.png")
# img2 = cv2.imread("map25.png")
# img = cv2.add(img, img2)
# cv2.imshow("q", img)
# mask_ = MaskWall(img)

# image, contours, hierachy = cv2.findContours(mask_, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# image = img

# cnt = contours[6]
# cv2.circle(img, (220, 170), 3, (255, 5, 55), -1)
# cv2.circle(img, (260, 160), 3, (255, 5, 55), -1)

# hull = cv2.convexHull(cnt)
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = numpy.int0(box)
# # cv2.drawContours(img,[box],0,(0,0,255),2)
# x, y, x2, y2, x3, y3, x4, y4 = ResizeRect(box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1])
# # print(x, y, x2, y2, x3, y3, x4, y4)
# print(IsCrossRect(x, y, x2, y2, x3, y3, x4, y4, 260, 160, 220, 170))

# ccwRect(x, y, x2, y2, x3, y3, x4, y4, 260, 160, 220, 170)

# cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
# cv2.circle(img, (x2, y2), 1, (0, 255, 255), -1)
# cv2.circle(img, (x3, y3), 1, (0, 255, 255), -1)
# cv2.circle(img, (x4, y4), 1, (0, 255, 255), -1)
# print(moveList)

# cv2.imshow("w", mask_)
# cv2.imshow("e", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img2 = cv2.imread("map25.png")
MoveList = []

while True:
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im2 = im[800:1070, 0:400]
    im2 = cv2.add(im2, img2)

    # y, x = MaskMap(im2)
    # if(y != -1 and x != -1):
    #     cv2.circle(im2, (x_, y_), 1, (255, 0, 0), -1)
    #     if isMoving == False:
    #         isMoving = FindWay(im2, x, y, moveList[0][0], moveList[0][1])
    #     else:

    cv2.imshow("2", im2)
    y, x = MaskMap(im2)

    # if len(MoveList) > 0:
    #     value = Move(x, y, MoveList[0][0][0], MoveList[0][0][1])
    #     if value == -1:
    #         del MoveList[0][0]

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(25) & 0xFF == ord('g'):
        im3 = im2
        # print(x, y)
        cv2.circle(im3, (120, 110), 1, (0, 255, 0), -1)
        if(y != -1 and x != -1):
            mask_ = MaskWall(im2)
            image, contours, hierachy = cv2.findContours(mask_, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            crossContour = None
            crossContourBox = []
            for idx, cnt in enumerate(contours):
                hull = cv2.convexHull(cnt)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = numpy.int0(box)
                cv2.drawContours(im3, [box], 0, (0, 0, 255), 2)
                if IsCrossRect(box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1], x, y, 120, 110) == True:
                    crossContour = box

            if crossContour is not None:
                x1, y1, x2, y2, x3, y3, x4, y4 = ResizeRect(crossContour[0][0], crossContour[0][1], crossContour[1][0], crossContour[1][1],
                                 crossContour[2][0], crossContour[2][1], crossContour[3][0], crossContour[3][1])
                # cv2.circle(im3, (x1, y1), 1, (0, 255, 255), -1)
                # cv2.circle(im3, (x2, y2), 1, (0, 255, 255), -1)
                # cv2.circle(im3, (x3, y3), 1, (0, 255, 255), -1)
                # cv2.circle(im3, (x4, y4), 1, (0, 255, 255), -1)
                # print(x, y, x2, y2, x3, y3, x4, y4)
                List = ccwRect(x, y, x2, y2, x3, y3, x4, y4, 120, 110, x, y)
                print(List)
                MoveList.insert(0, (120, 110))
                MoveList.insert(0, List)
                cv2.circle(im3, (List[0][0], List[0][1]), 1, (0, 255, 0), -1)

                # cv2.drawContours(img,[box],0,(0,0,255),2)
            cv2.imshow("3", im3)

        # cv2.destroyAllWindows()
        # break