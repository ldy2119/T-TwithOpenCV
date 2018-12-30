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

from PIL import ImageGrab
from PIL import Image
import numpy
import time
import cv2
from ctypes import windll
import pytesseract
import time

user32 = windll.user32
user32.SetProcessDPIAware()
 
while(True):
    image = ImageGrab.grab(bbox=(370, 970, 580, 1070))
    printScreen = numpy.array(image)
    printScreen = cv2.imread("asdf\\1546182102.png")
    # printScreen = cv2.imread("asdf.PNG")
    # printScreen = cv2.cvtColor(printScreen, cv2.COLOR_BGR2RGB)
    printScreen = cv2.cvtColor(printScreen, cv2.COLOR_RGB2GRAY)
    ret, printScreen = cv2.threshold(printScreen, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('window', printScreen)
    img_new = Image.fromarray(printScreen)
    text = pytesseract.image_to_string(img_new)
    print(text)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
