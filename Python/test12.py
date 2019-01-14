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

#북서쪽으로 이동
def MoveNW():
    keyboard.press('w')
    keyboard.press('a')

#북동쪽으로 이동
def MoveNE():
    keyboard.press('w')
    keyboard.press('d')

#남서쪽으로 이동
def MoveSW():
    keyboard.press('s')
    keyboard.press('a')

#남동쪽으로 이동
def MoveSE():
    keyboard.press('s')
    keyboard.press('d')

#집으로 이동
# def ReturnHome():
#     keyboard.press('r')
#     time.sleep(2)
#     keyboard.press(Key.space)
#     ReleaseAll()

def FindWall(maskMap, x, y, img, preindex):
    point_nw = [(x + 5, y - 9), (x, y - 6), (x - 9, y), (x - 13, y + 3)]
    point_ne = [(x - 5, y - 9), (x, y - 6), (x + 9, y), (x + 13, y + 3)]
    point_sw = [(x + 5, y + 9), (x, y + 6), (x - 9, y), (x - 13, y - 3)]
    point_se = [(x - 5, y + 9), (x, y + 6), (x + 9, y), (x + 13, y - 3)]
    point = [point_nw, point_ne, point_sw, point_se]

    point = [point[(preindex + 1) % 4], point[(preindex + 2) % 4], point[(preindex + 3) % 4], point[(preindex + 4) % 4]]
    # print((preindex + 1) %4, (preindex + 2)%4,( preindex + 3)%4, (preindex + 4)%4)
    max = 0
    maxindex = -1
    for idx, arr in enumerate(point):
        tmp = 0
        for y, x in arr:
            # print((idx + preindex + 1) % 4)
            # print(x, y)
            tmp = Findaround(maskMap, x, y)
            if tmp > 0:
                # print(x, y)
                tmp += 1
                cv2.circle(img, (y, x), 1, (255, 0, 255), -1)
        if tmp > max:
            max = tmp
            maxindex = (idx + preindex + 1) % 4

    # print(point[maxindex])
    print(maxindex)
    return maxindex

# im = cv2.imread("map.png")

# y,x = MaskMap(im)
# print(x, y)
# cv2.imshow("q", im)
# cv2.waitKey(0)
index_ = 0
preindex = -1
while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im2 = im[800:1070, 0:400]

    y, x = MaskMap(im2)
    if(y != -1 and x != -1):
        mask_ = MaskWall(im2)
        index_ = FindWall(mask_, x, y, im2, preindex)
        print(index_)
        if index_ == 0:
            ReleaseAll()
            MoveNE()
        if index_ == 1:
            ReleaseAll()
            MoveSE()
        if index_ == 2:
            ReleaseAll()
            MoveNW()
        if index_ == 3:
            MoveSW()
        preindex = index_

    cv2.imshow("2", im2)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break