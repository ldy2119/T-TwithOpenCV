from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll
from pynput.keyboard import Key, Controller
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

#마스크 이미지
mapMaskImage = cv2.imread("map25.png")

#플레이어의 좌표
x = y = 0

#플레이어 자원
food = 0

#플레이어 행동
scout = 0
build = 1
farm = 2
attack = 3

#현재 행동상태
now_Player_Type = scout

#현재 진행방향
stop = 4
nw = 0
ne = 1
sw = 2
se = 3

#키보드 입력을 받는다
keyboard = Controller()

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

#벽의 형태를 만든다(마스킹한다)
def MaskWall(im):
    lower_pl = numpy.array([3, 0, 2])
    upper_pl = numpy.array([20, 13, 18])
    mask_pl = cv2.inRange(im, lower_pl, upper_pl)
    # cv2.imshow("wal", mask_pl)
    return mask_pl

#플레이어 위치를 찾는다(마스킹한다)
def MaskPl(im):
    dst = cv2.add(im, mapMaskImage)
    mask_pl = cv2.inRange(dst, lower_pl, upper_pl)
    # adf = cv2.bitwise_and(im, im, mask = mask_pl)
    # cv2.imshow("e", adf)
    return mask_pl

#플레이어 주변 공간만 보이게 한다
def MakeROI(im, y, x):
    minX = x - 50 if x >= 50 else 0
    maxX = x + 50 if x <= 220 else 270
    minY = y - 50 if y >= 50 else 0
    maxY = y + 50 if y <= 350 else 0
    # print(minX, maxX, minY, maxY)
    # print(x, y)
    return im[minX:maxX, minY:maxY]

#점 주변에 벽이 있는지?
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
            # print(x, y)
            return 1
    return 0

#모든 키를 누르지 않는다(손에서 뗀다)
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
def ReturnHome():
    keyboard.press('r')
    time.sleep(2)
    keyboard.press(Key.space)
    ReleaseAll()

#벽을 찾는다
# def FindWall(maskMap, y, x):
#     #근처에 벽이 있는지?
#     #동, 서, 남, 북 어느 방향에 벽이 있는지?
    
#     point_nw = [(x + 5, y - 9), (x, y + 6), (x - 9, y), (x - 13, y + 3)]
#     point_ne = [(x - 5, y - 9), (x, y + 6), (x + 9, y), (x + 13, y + 3)]
#     point_sw = [(x + 5, y + 9), (x, y - 6), (x - 9, y), (x - 13, y - 3)]
#     point_se = [(x - 5, y + 9), (x, y - 6), (x + 9, y), (x + 13, y - 3)]
#     point = [point_nw, point_ne, point_sw, point_se]
#     cv2.imshow("m", maskMap)
#     max = -1
#     maxindex = -1
#     for idx, arr in enumerate(point):
#         tmp = 0
#         for x, y in arr:
#             tmp += Findaround(maskMap, x, y)
#             # if maskMap[x, y] > 0:
#                 # print(x, y)
#             #     tmp += 1
#         # print(tmp)
#         if tmp >= 2:
#             if tmp > max:
#                     max = tmp
#                     maxindex = idx
#         # cv2.imshow("asdf", maskMap)
#         # cv2.line(img1_fg, arr[0], arr[3], (255, 255, 255), 1)

#     return maxindex

#벽을 찾는다
def FindWall(maskMap, x, y):
    #근처에 벽이 있는지?
    #동, 서, 남, 북 어느 방향에 벽이 있는지?
    point_nw = [(x + 5, y - 9), (x, y + 6), (x - 9, y), (x - 13, y + 3)]
    point_ne = [(x - 5, y - 9), (x, y + 6), (x + 9, y), (x + 13, y + 3)]
    point_sw = [(x + 5, y + 9), (x, y - 6), (x - 9, y), (x - 13, y - 3)]
    point_se = [(x - 5, y + 9), (x, y - 6), (x + 9, y), (x + 13, y - 3)]
    point = [point_nw, point_ne, point_sw, point_se]
    max = 0
    maxindex = -1
    for idx, arr in enumerate(point):
        tmp = 0
        for y, x in arr:
            tmp = Findaround(maskMap, x, y)
            if tmp > 0:
                print(x, y)
                tmp += 1
        if tmp >= 2:
            if tmp > max:
                print(x, y)
                max = tmp
                maxindex = idx
    return maxindex

#집의 위치를 알아낸다
def FindHome(im):
    method = eval('cv2.TM_CCOEFF')

    res = cv2.matchTemplate(im,template,method)
    min_val,max_val,min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = 0

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return int(top_left[0] + w/2), int(top_left[1] + h/2)

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)
 
while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im1 = im[970:1070, 370:580]
    im2 = im[800:1070, 0:400]
    food = GetFood(im1)
    mask_pl = MaskPl(im2)
    x, y = FindPlayer(mask_pl)

    if x != -1 and y != -1:
        x, y = y, x
        # mask_wall = MakeROI(im2, x, y)
        mask_wall = im2
        mask_wall = MaskWall(mask_wall)
        index = FindWall(mask_wall, x, y)
        
        
        print(index)
    cv2.imshow("2", im2)
    # cv2.imwrite("map11111.png", im2)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    #코드(였던 것)
    # cv2.imshow("1", im1)

    # image = ImageGrab.grab(bbox=(370, 970, 580, 1070))
    # printScreen = numpy.array(image)
    # image2 = ImageGrab.grab(bbox=(0, 800, 400, 1070))
    # print2 = numpy.array(image2)
    # print2 = cv2.cvtColor(print2, cv2.COLOR_BGR2RGB)
    # printScreen = cv2.cvtColor(printScreen, cv2.COLOR_BGR2RGB)
    # printScreen = cv2.cvtColor(printScreen, cv2.COLOR_RGB2GRAY)
    # ret, printScreen = cv2.threshold(printScreen, 90, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('win2', print2)
    # img_new = Image.fromarray(printScreen)
    # text = pytesseract.image_to_string(img_new)
    # print(text)
    # print(x, y)