import cv2
import numpy as np
from pynput.keyboard import Key, Controller

img = cv2.imread("map.png", cv2.COLOR_BGR2GRAY)
img2 = img[175:180, 260:260]
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


cv2.imshow("", img)
print(img2)
print(img2[3,3])
cv2.waitKey(0)
cv2.destroyAllWindows()

#13 0 5  벽

# 41 41 195  농장

#54 56 46  안개

#39 39 187 풍차

#28 28 193 플레이어

# 20 20 187

# 35 35 200