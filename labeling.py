import cv2
import os

def main():
    dir = "/home/nguyendat/PycharmProjects/SignRecognition/SignData/Sign"
    labelsFile = open("/home/nguyendat/PycharmProjects/SignRecognition/SignData/labels.csv", "a")
    labelsFile.write("label\n")
    i = 0
    label = ""
    for i in range(0, 223):
        fname = dir + str(i) + ".png"
        pic = cv2.imread(fname)
        cv2.imshow('pic' + str(i), pic)
        cv2.waitKey(0)
        label = input("Enter label: ")
        labelsFile.write(label + '\n')
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

