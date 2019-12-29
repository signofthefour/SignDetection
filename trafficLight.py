import cv2
import numpy as np


if __name__ == '__main__':
    path = '/home/nguyendat/PycharmProjects/SignRecognition/TrafficLight/trafficLight6.png'
    mapCap = cv2.VideoCapture('/home/nguyendat/PycharmProjects/SignRecognition/public4.mp4')
    while (True):
        ret, img = mapCap.read()
        frame = img[0:int(len(img[0])/2), 0:]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_light = np.array([5, 180, 210])
        upper_light = np.array([20, 200, 230])

        mask = cv2.inRange(hsv, lower_light, upper_light)

        # kernel = np.ones((3, 3), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=1)
        count = 0
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        try:
            cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
            print('something')
        except:
            print('NONE')

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    mapCap.release()
    cv2.destroyAllWindows()