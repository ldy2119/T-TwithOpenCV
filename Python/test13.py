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
    # adf = cv2.bitwise_and(im, im, mask = mask_pl)
    # cv2.imshow("e", adf)

    return FindPlayer(mask_pl)

def MaskWall(im):
    lower_pl = numpy.array([3, 0, 2])
    upper_pl = numpy.array([20, 13, 19])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)
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

#경로 설정
moveList = [(220, 100)]

#경로 설정
isMoving = False

while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im2 = im[800:1070, 0:400]

    y, x = MaskMap(im2)
    if(y != -1 and x != -1):
        # cv2.circle(im2, (x_, y_), 1, (255, 0, 0), -1)
        if isMoving == False:
            isMoving = FindWay(im2, x, y, moveList[0][0], moveList[0][1])
        else:
            if len(moveList) > 0:
                value = Move(x, y, moveList[0][0], moveList[0][1])
                if value == -1:
                    del moveList[0]


    cv2.imshow("2", im2)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break