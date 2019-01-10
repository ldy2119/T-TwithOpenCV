from pynput.keyboard import Key, Controller
import time

def ReleaseAll():
    keyboard.release('r')
    keyboard.release('a')
    keyboard.release('s')
    keyboard.release('d')
    keyboard.release('w')
    keyboard.release(Key.space)

keyboard = Controller()

keyboard.press('r')
time.sleep(1)
ReleaseAll()