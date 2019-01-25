#마우스 클릭 구현


from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll
from pynput.mouse import Button, Controller as MouseCtrl
from pynput.keyboard import Key, Controller as KeyCtrl
import time

user32 = windll.user32
user32.SetProcessDPIAware()
keyboard = KeyCtrl()

mouse = MouseCtrl()

def FindPlayer(mask_pl):
    arr = numpy.where(mask_pl != 0)
    if len(arr[0]) != 0 or len(arr[1]) != 0:
        return arr[0][0], arr[1][0]
    return -1, -1

def MaskPlayer(im):
    lower_pl = numpy.array([15, 15, 175])
    upper_pl = numpy.array([35, 35, 190])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)

    return FindPlayer(mask_pl)

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

def MouseClick():
    mouse.press(Button.left)
    time.sleep(0.1)
    mouse.release(Button.left)

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

img = cv2.imread("map8.png")

BuildFlag = False
MoveList = []
while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im1 = im[970:1070, 370:580]
    im2 = im[800:1070, 0:400]
    
    if BuildFlag == True and len(MoveList) == 0:
        BuildFlag = False

    elif BuildFlag == True and len(MoveList) > 0:
        y, x = MaskPlayer(im2)
        value = Move(x, y, MoveList[0][0], MoveList[0][1])
        MouseClick()

        print(x, y, MoveList)

        if value == -1:
            del MoveList[0]

    elif BuildFlag == False:
        y, x = MaskEnemy(im2)
        MoveList.insert(0, (x, y))
        BuildFlag = True

    cv2.imshow("e", im2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break