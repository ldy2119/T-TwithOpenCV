#스크린샷 찍기

from PIL import ImageGrab
from PIL import Image
import numpy
import cv2
from ctypes import windll

user32 = windll.user32
user32.SetProcessDPIAware()

while(True):
    im = ImageGrab.grab()
    im = numpy.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im1 = im[970:1070, 370:580]
    im2 = im[800:1070, 0:400]

    cv2.imshow("2", im2)
    cv2.imwrite("map11111.png", im2)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break