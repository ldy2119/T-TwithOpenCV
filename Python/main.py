from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll
from pynput.keyboard import Key, Controller as KeyCtrl
from pynput.mouse import Button, Controller as MouseCtrl

import time

user32 = windll.user32
user32.SetProcessDPIAware()

#학습된 정보를 들고 온다
samples = numpy.loadtxt('generalsamples.data',numpy.float32)
responses = numpy.loadtxt('generalresponses.data',numpy.float32)
responses = responses.reshape((responses.size,1))

#마스크 관련 값
lower_pl = numpy.array([15, 15, 175])
upper_pl = numpy.array([35, 35, 190])

lower_wall = numpy.array([3, 0, 2])
upper_wall = numpy.array([20, 13, 18])

#마스크 이미지
mapMaskImage = cv2.imread("map25.png")

#플레이어의 좌표
x = y = 0

#플레이어 자원
food = 0

#현재 진행방향
stop = 4
nw = 0
ne = 1
sw = 2
se = 3

#키보드 입력을 받는다
keyboard = KeyCtrl()
mouse = MouseCtrl()

#농장 및 병영 건설 위치
arr = [(-14, 0), (14, 0), (0, -10), (0, 10), (-7, -5), (7, -5), (-7, 5), (7, 5)]
arr2 = [(0, 20), (7, 15), (14, 10)]

#집 모양을 알아낸다
template = cv2.imread("map3.png")
h,w,d = template.shape #높이, 너비, 색상 갯수, d는 의미 없음

#식량을 가지고 온다
def GetFood(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    food = 0
    pow = 1

    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h == 70:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = numpy.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = int((results[0][0]))
                if type(string) == type(food):
                    food = food + pow * string
                    pow = pow * 10
    return food

#플레이어의 좌표를 찾는다
def FindPlayer(mask_pl):
    arr = numpy.where(mask_pl != 0)
    if len(arr[0]) != 0 or len(arr[1]) != 0:
        return arr[0][0], arr[1][0]
    return -1, -1

#플레이어 위치를 찾는다(마스킹한다)
def MaskPlayer(im):
    lower_pl = numpy.array([15, 15, 175])
    upper_pl = numpy.array([35, 35, 190])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)

    return FindPlayer(mask_pl)

#벽의 형태를 만든다(마스킹한다)
def MaskWall(im):
    mask_pl = cv2.inRange(im, lower_wall, upper_wall)
    # cv2.imshow("wal", mask_pl)
    return mask_pl

def FindEnemy(mask_pl):
    arr = numpy.where(mask_pl != 0)
    if len(arr[0]) != 0 or len(arr[1]) != 0:
        return arr[0][0], arr[1][0]
    return -1, -1

def MaskEnemy(im):
    lower_pl = numpy.array([190, 110, 25])
    upper_pl = numpy.array([220, 130, 35])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)

    return FindPlayer(mask_pl)

#모든 키를 누르지 않는다(손에서 뗀다)
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

#집으로 이동
def ReturnHome():
    keyboard.press('r')
    time.sleep(2)
    keyboard.press(Key.space)
    ReleaseAll()

def Move(x, y, x_, y_):
    value = -1

    ReleaseAll()
    if abs(x_ - x) > 3:
        if x < x_:
            MoveE()
        elif x > x_:
            MoveW()
        value = 1
    
    if abs(y_ - y) > 3:
        if y < y_:
            MoveS()
        elif y > y_:
            MoveN()
        value = 1
    return value

def MouseClick():
    mouse.press(Button.left)
    time.sleep(0.1)
    mouse.release(Button.left)

#######
#농장을 짓는다

def FindFarm(maskMap, x, y):
    array = []
    # cv2.imshow("q", maskMap)
    count = 0
    for x_, y_ in arr:
        tmp = Findaround(maskMap, x + x_, y + y_)
        if tmp == 1:
            array.append(1)
            count += 1
            if count >= 6:
                return []
        else:
            array.append(0)
    return array

def BuildFarm(img, x, y):
    lower_pl = numpy.array([15, 15, 175])
    upper_pl = numpy.array([50, 50, 210])

    mask_ = cv2.inRange(img, lower_pl, upper_pl)

    array = FindFarm(mask_, x, y)
    for idx, tmp in enumerate(array):
        if array[idx] == 0:
            cv2.circle(img, (arr[idx][0] + x, arr[idx][1] + y), 1, (255, 0, 0), -1)
            return arr[idx][0] + x, arr[idx][1] + y
    return -1, -1

#######
#병영을 짓는다

def FindBarrack(maskMap, x, y):
    array = []
    count = 0
    cv2.imshow("asdf", maskMap)
    for x_, y_ in arr2:
        tmp = Findaround(maskMap, x + x_, y + y_)
        if tmp == 1:
            array.append(1)
            count += 1
            if count >= 3:
                return []
        else:
            array.append(0)
    return array

def BuildBarrak(img, x, y):
    lower_pl = numpy.array([120, 120, 200])
    upper_pl = numpy.array([135, 135, 215])

    mask_ = cv2.inRange(img, lower_pl, upper_pl)

    array = FindBarrack(mask_, x, y)
    for idx, tmp in enumerate(array):
        if array[idx] == 0:
            cv2.circle(img, (arr2[idx][0] + x, arr2[idx][1] + y), 1, (255, 0, 0), -1)
            return arr2[idx][0] + x, arr2[idx][1] + y
    return -1, -1

#######

def Build():
    keyboard.press(Key.space)
    time.sleep(0.5)
    keyboard.release(Key.space)
    
def Findaround(maskMap, y, x):
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

def FindWindmill(img):
    template = cv2.imread("map7.png")
    h,w,d = template.shape

    method = eval('cv2.TM_CCOEFF_NORMED')

    res = cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = 0

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    x = top_left[0] + 7
    y = top_left[1] + 4
    print(x, y)
    return x, y

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
    if x == x2 or x == x3 or x == x4:
        return -1, -1, -1, -1, -1, -1, -1, -1

    if x > x3:
        x, y, x3, y3 = x3, y3, x, y
    if x2 > x4:
        x2, y2, x4, y4 = x4, y4, x2, y2
    
    a, b = FindSlope(x, y, x3, y3)
    a2, b2 = FindSlope(x2, y2, x4, y4)
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
    arr.sort(key = lambda obj : obj[1], reverse = True)
    # print(arr)
    ccwtemp = []
    for tmp in arr:
        ccwtemp.append(ccw(curx, cury, tmp[0], tmp[1], dirx, diry))

    for idx, tmp in enumerate(ccwtemp):
        if ccwtemp[idx] == -1:
            moveList.insert(0, (arr[idx][0], arr[idx][1]))
    return moveList

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

MillX = -1
MillY = -1

MoveList = []
BuildFlag = False
MouseFlag = False

while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im1 = im[970:1070, 370:580]
    im2 = im[800:1070, 0:400]
    food = GetFood(im1)
    x, y = MaskPlayer(im2)

    if MillX == -1 or MillY == -1:
        MillX, MillY = FindWindmill(im2)
    
    if BuildFlag == True and len(MoveList) == 0:
        if not MouseFlag:
            Build()
        BuildFlag = False
        MouseFlag = False

    elif BuildFlag == True and len(MoveList) > 0:
        y, x = MaskPlayer(im2)
        value = Move(x, y, MoveList[0][0], MoveList[0][1])
        print(x, y, MoveList)

        if value == -1:
            del MoveList[0]

    elif food >= 60 and BuildFlag == False:
        y, x = MaskPlayer(im2)
        x_, y_ = BuildFarm(im2, MillX, MillY)
        if x_ == -1 or y_ == -1:
            x_, y_ = BuildBarrak(im2, MillX, MillY)
            if x_ == -1 or y_ == -1:
                y_, x_ = MaskEnemy(im2)
                MouseFlag = True
        MoveList.insert(0, (x_, y_))
        BuildFlag = True

    cv2.imshow("e", im2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break