#농장 찾기(수정 바람)


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

arr = [(-14, 0), (14, 0), (0, -10), (0, 10), (-7, -5), (7, -5), (-7, 5), (7, 5)]


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

def FindFarm(maskMap, x, y):
    arr2 = []
    # cv2.imshow("q", maskMap)
    for x_, y_ in arr:
        tmp = Findaround(maskMap, x + x_, y + y_)
        if tmp == 1:
            arr2.append(1)
        else:
            arr2.append(0)
    return arr2

def BuildFarm(img, x, y):
    lower_pl = numpy.array([15, 15, 175])
    upper_pl = numpy.array([50, 50, 210])

    mask_ = cv2.inRange(img, lower_pl, upper_pl)

    arr2 = FindFarm(mask_, x, y)
    for idx, tmp in enumerate(arr2):
        if arr2[idx] == 0:
            cv2.circle(img, (arr[idx][0] + x, arr[idx][1] + y), 1, (255, 0, 0), -1)
            return

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

    methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

    method = eval('cv2.TM_CCOEFF_NORMED')

    res = cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = 0

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    print(int(top_left[0] + 7), int(top_left[1] + 5))
    x = top_left[0] + 7
    y = top_left[1] + 5
    return x, y

def Build():
    keyboard.press(Key.space)
    time.sleep(0.5)
    keyboard.release(Key.space)

# x, y = 205, 56
# img2 = cv2.imread("map.png")
# # 이제 농장이 있는지 확인해야 함!!!
# # cv2.circle(img2, (x, y), 1, (255,0,255), -1)
# cv2.circle(img2, (x-14, y), 1, (255,0,255), -1)
# cv2.circle(img2, (x+14, y), 1, (255,0,255), -1)
# cv2.circle(img2, (x, y+10), 1, (255,0,255), -1)
# cv2.circle(img2, (x, y-10), 1, (255,0,255), -1)
# cv2.circle(img2, (x+7, y+5), 1, (255,0,255), -1)
# cv2.circle(img2, (x+7, y-5), 1, (255,0,255), -1)
# cv2.circle(img2, (x-7, y+5), 1, (255,0,255), -1)
# cv2.circle(img2, (x-7, y-5), 1, (255,0,255), -1)


# cv2.circle(img2, (x, y - 20), 1, (255,0,255), -1)
# cv2.circle(img2, (x-7, y-15), 1, (255,0,255), -1)
# cv2.circle(img2, (x-14, y-10), 1, (255,0,255), -1)

# cv2.circle(img2, (x, y), 1, (255,0,255), -1)
# # cv2.circle(img2, (x, y), 1, (255,0,255), -1)
# # cv2.circle(img2, (int(top_left[0] + w/2) + 15, int(top_left[1] + h/2)), 1, (255,0,255), -1)
# # cv2.circle(img2, (int(top_left[0] + w/2) - 15, int(top_left[1] + h/2)), 1, (255,0,255), -1)
# # cv2.circle(img2, (int(top_left[0] + w/2), int(top_left[1] + h/2) + 10), 1, (255,0,255), -1)
# # cv2.circle(img2, (int(top_left[0] + w/2), int(top_left[1] + h/2) - 10), 1, (255,0,255), -1)
# # cv2.circle(img2, (int(top_left[0] + w/2) + 10, int(top_left[1] + h/2) + 5), 1, (255,0,255), -1)
# # cv2.circle(img2, (int(top_left[0] + w/2) - 10, int(top_left[1] + h/2) + 5), 1, (255,0,255), -1)
# # cv2.circle(img2, (int(top_left[0] + w/2) + 10, int(top_left[1] + h/2) - 5), 1, (255,0,255), -1)
# # cv2.circle(img2, (int(top_left[0] + w/2) - 10, int(top_left[1] + h/2) - 5), 1, (255,0,255), -1)

# lower_pl = numpy.array([15, 15, 175])
# upper_pl = numpy.array([50, 50, 210])

# mask_ = cv2.inRange(img2, lower_pl, upper_pl)


# # print(mask_[])
# print(Findaround(mask_, x, y - 10))



# cv2.imshow("a", img2)
# cv2.waitKey(0)

MillX = -1
MillY = -1

while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im1 = im[970:1070, 370:580]
    im2 = im[800:1070, 0:400]
    food = GetFood(im1)
    
    # if cv2.waitKey(25) & 0xFF == ord('g'):
    # print(food)
    if MillX == -1 or MillY == -1:
        MillX, MillY = FindWindmill(im2)
    if food >= 60:
        BuildFarm(im2, MillX, MillY)

    cv2.imshow("e", im2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break