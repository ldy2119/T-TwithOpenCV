#배럭 짓기

from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll
from pynput.keyboard import Key, Controller
import time

user32 = windll.user32
user32.SetProcessDPIAware()
keyboard = Controller()

#학습된 정보를 들고 온다
samples = numpy.loadtxt('generalsamples.data',numpy.float32)
responses = numpy.loadtxt('generalresponses.data',numpy.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

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

def Build():
    keyboard.press(Key.space)
    time.sleep(0.5)
    keyboard.release(Key.space)

while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im1 = im[970:1070, 370:580]
    im2 = im[800:1070, 0:400]
    food = GetFood(im1)
    
    # if cv2.waitKey(25) & 0xFF == ord('g'):
    # print(food)
    if food >= 60:
        Build()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


#193 118 118