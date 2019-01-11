from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll

def FindFarm(maskMap, x, y):
    arr = [(-14, 0), (14, 0), (0, -10), (0, 10), (-7, -5), (7, -5), (-7, 5), (7, 5)]
    arr2 = []
    for x_, y_ in arr:
        tmp = Findaround(maskMap, x + x_, y + y_)
        if tmp == 1:
            arr2.append(1)
        else:
            arr2.append(0)
    return arr2

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

img2 = cv2.imread("map2.png")
template = cv2.imread("map3.png")
h,w,d = template.shape

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

img = img2.copy()
method = eval('cv2.TM_CCOEFF')

res = cv2.matchTemplate(img,template,method)
min_val,max_val,min_loc, max_loc = cv2.minMaxLoc(res)
top_left = 0

if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
print(int(top_left[0] + 6), int(top_left[1] + 5))
x = top_left[0] + 6
y = top_left[1] + 5

# 이제 농장이 있는지 확인해야 함!!!
# cv2.circle(img2, (x, y), 1, (255,0,255), -1)
cv2.circle(img2, (x-14, y), 1, (255,0,255), -1)
cv2.circle(img2, (x+14, y), 1, (255,0,255), -1)
cv2.circle(img2, (x, y+10), 1, (255,0,255), -1)
cv2.circle(img2, (x, y-10), 1, (255,0,255), -1)
cv2.circle(img2, (x+7, y+5), 1, (255,0,255), -1)
cv2.circle(img2, (x+7, y-5), 1, (255,0,255), -1)
cv2.circle(img2, (x-7, y+5), 1, (255,0,255), -1)
cv2.circle(img2, (x-7, y-5), 1, (255,0,255), -1)

cv2.circle(img2, (x, y), 1, (255,0,255), -1)
# cv2.circle(img2, (x, y), 1, (255,0,255), -1)
# cv2.circle(img2, (int(top_left[0] + w/2) + 15, int(top_left[1] + h/2)), 1, (255,0,255), -1)
# cv2.circle(img2, (int(top_left[0] + w/2) - 15, int(top_left[1] + h/2)), 1, (255,0,255), -1)
# cv2.circle(img2, (int(top_left[0] + w/2), int(top_left[1] + h/2) + 10), 1, (255,0,255), -1)
# cv2.circle(img2, (int(top_left[0] + w/2), int(top_left[1] + h/2) - 10), 1, (255,0,255), -1)
# cv2.circle(img2, (int(top_left[0] + w/2) + 10, int(top_left[1] + h/2) + 5), 1, (255,0,255), -1)
# cv2.circle(img2, (int(top_left[0] + w/2) - 10, int(top_left[1] + h/2) + 5), 1, (255,0,255), -1)
# cv2.circle(img2, (int(top_left[0] + w/2) + 10, int(top_left[1] + h/2) - 5), 1, (255,0,255), -1)
# cv2.circle(img2, (int(top_left[0] + w/2) - 10, int(top_left[1] + h/2) - 5), 1, (255,0,255), -1)

print(top_left)

lower_pl = numpy.array([15, 15, 175])
upper_pl = numpy.array([50, 50, 210])

mask_ = cv2.inRange(img2, lower_pl, upper_pl)
cv2.imshow("e", mask_[140:170, 260:280])
# print(mask_[])
print(Findaround(mask_, y + 10, x))


cv2.imshow("a", img2)
cv2.waitKey(0)
