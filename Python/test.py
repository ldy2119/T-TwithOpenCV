# from PIL import ImageGrab
# import os
# import time
 
# def screenGrab():
#     box = ()
#     im = ImageGrab.grab()
#     im.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) +
# '.png', 'PNG')
 
# def main():
#     screenGrab()
 
# if __name__ == '__main__':
#     main()

# from PIL import ImageGrab
# from PIL import Image
# import numpy
# import time
# import cv2
# from ctypes import windll
# import pytesseract
# import time

# user32 = windll.user32
# user32.SetProcessDPIAware()
 
# while(True):
#     image = ImageGrab.grab(bbox=(370, 970, 580, 1070))
#     printScreen = numpy.array(image)
#     # printScreen = cv2.imread("asdf\\1546182102.png")
#     # printScreen = cv2.imread("asdf.PNG")
#     # printScreen = cv2.cvtColor(printScreen, cv2.COLOR_BGR2RGB)
#     printScreen = cv2.cvtColor(printScreen, cv2.COLOR_RGB2GRAY)
#     # ret, printScreen = cv2.threshold(printScreen, 90, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow('window', printScreen)
#     img_new = Image.fromarray(printScreen)
#     img_new.save(str(int(time.time())) + ".png", "PNG")
#     # text = pytesseract.image_to_string(img_new)
#     # print(text)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

import sys

import numpy as np
import cv2

im = cv2.imread('asdf.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

image, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print ("training complete")

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)