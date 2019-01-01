import cv2
import numpy as np

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

im = cv2.imread('1546182101.png')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

image, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h == 70:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            string = str(int((results[0][0])))

cv2.imshow('im',im)
cv2.waitKey(0)

#-*- coding: utf-8 -*-
# import cv2
# import numpy as np
# import glob
# import sys

# FNAME = 'digits.npz'

# def machineLearning():
#     img = cv2.imread('asdf.png')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
#     x = np.array(cells)
#     train = x[:,:].reshape(-1,400).astype(np.float32)

#     k = np.arange(10)
#     train_labels = np.repeat(k,500)[:,np.newaxis]

#     np.savez(FNAME,train=train,train_labels = train_labels)

# def resize20(pimg):
#     img = cv2.imread(pimg)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     grayResize = cv2.resize(gray,(20,20))
#     ret, thresh = cv2.threshold(grayResize, 125, 255,cv2.THRESH_BINARY_INV)

#     cv2.imshow('num',thresh)
#     return thresh.reshape(-1,400).astype(np.float32)

# def loadTrainData(fname):
#     with np.load(fname) as data:
#         train = data['train']
#         train_labels = data['train_labels']

#     return train, train_labels

# def checkDigit(test, train, train_labels):
#     knn = cv2.ml.KNearest_create()
#     knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

#     ret, result, neighbours, dist = knn.findNearest(test, k=5)

#     return result

# if __name__ == '__main__':
#     if len(sys.argv) == 1:
#         print ('option : train or test')
#         exit(1)
#     elif sys.argv[1] == 'train':
#         machineLearning()
#     elif sys.argv[1] == 'test':
#         train, train_labels = loadTrainData(FNAME)

#         saveNpz = False
#         for fname in glob.glob('images/num*.png'):
#             test = resize20(fname)
#             result = checkDigit(test, train, train_labels)

#             print (result)

#             k = cv2.waitKey(0)

#             if k > 47 and k<58:
#                 saveNpz = True
#                 train = np.append(train, test, axis=0)
#                 newLabel = np.array(int(chr(k))).reshape(-1,1)
#                 train_labels = np.append(train_labels, newLabel,axis=0)


#         cv2.destroyAllWindows()
#         if saveNpz:
#             np.savez(FNAME,train=train, train_labels=train_labels)
#     else:
#         print ('unknow option')