from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll

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

def MaskWall(im):
    # print()

#맵 형태를 만든다(주변 잡음을 제거)(플레이어만 남긴다)
def MaskMap(im):
    dst = cv2.add(im, mapMaskImage)
    mask_pl = cv2.inRange(dst, lower_pl, upper_pl)
    adf = cv2.bitwise_and(im, im, mask = mask_pl)
    # cv2.imshow("e", adf)
    return FindPlayer(mask_pl)

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)
 
while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im1 = im[970:1070, 370:580]
    cv2.imshow("1", im1)
    im2 = im[800:1070, 0:400]
    cv2.imshow("2", im2)
    # image = ImageGrab.grab(bbox=(370, 970, 580, 1070))
    # printScreen = numpy.array(image)
    # image2 = ImageGrab.grab(bbox=(0, 800, 400, 1070))
    # print2 = numpy.array(image2)
    # print2 = cv2.cvtColor(print2, cv2.COLOR_BGR2RGB)
    # printScreen = cv2.cvtColor(printScreen, cv2.COLOR_BGR2RGB)
    # printScreen = cv2.cvtColor(printScreen, cv2.COLOR_RGB2GRAY)
    # ret, printScreen = cv2.threshold(printScreen, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('window', im2)
    # cv2.imshow('win2', print2)
    # img_new = Image.fromarray(printScreen)
    # text = pytesseract.image_to_string(img_new)
    food = GetFood(im1)
    x, y = MaskMap(im2)
    # print(text)
    print(x, y)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
