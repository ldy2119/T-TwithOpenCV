from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll

img2 = cv2.imread("map.png")
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
print(int(top_left[0] + h/2), int(top_left[1] + w/2))

cv2.circle(img2, (int(top_left[0] + w/2), int(top_left[1] + h/2)), 1, (255,0,255), -1)
cv2.circle(img2, (int(top_left[0] + w/2) + 15, int(top_left[1] + h/2)), 1, (255,0,255), -1)
cv2.circle(img2, (int(top_left[0] + w/2) - 15, int(top_left[1] + h/2)), 1, (255,0,255), -1)
cv2.circle(img2, (int(top_left[0] + w/2), int(top_left[1] + h/2) + 10), 1, (255,0,255), -1)
cv2.circle(img2, (int(top_left[0] + w/2), int(top_left[1] + h/2) - 10), 1, (255,0,255), -1)
cv2.circle(img2, (int(top_left[0] + w/2) + 10, int(top_left[1] + h/2) + 5), 1, (255,0,255), -1)
cv2.circle(img2, (int(top_left[0] + w/2) - 10, int(top_left[1] + h/2) + 5), 1, (255,0,255), -1)
cv2.circle(img2, (int(top_left[0] + w/2) + 10, int(top_left[1] + h/2) + 5), 1, (255,0,255), -1)
cv2.circle(img2, (int(top_left[0] + w/2) - 10, int(top_left[1] + h/2) - 5), 1, (255,0,255), -1)


cv2.imshow("a", img2)
cv2.waitKey(0)
