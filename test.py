import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from trainSignDetect import detect_sign


svclassifier = pickle.load(open('signDetect', 'rb'))
mapCap = cv2.VideoCapture('/home/nguyendat/PycharmProjects/SignRecognition/public4.mp4')
counter = 0
while (True):
    ret, frame = mapCap.read()
    img_save = frame
    size, sign = detect_sign(frame)
    if sign is not None:
        command = svclassifier.predict(np.reshape(sign, (1, -1)))
        if 30 > size[0] > 10 and 30 > size[1] > 10:
            print(command)
        # print(command)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
mapCap.release()
cv2.destroyAllWindows()