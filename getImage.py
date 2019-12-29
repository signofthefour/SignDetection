import cv2

path = '/home/nguyendat/PycharmProjects/SignRecognition'
folder = '/home/nguyendat/PycharmProjects/SignRecognition/MapImage'

cap = cv2.VideoCapture(path + '/public4.mp4')
count = 0
num = 255
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    count += 1
    if count == 100:
        cv2.imwrite(folder + '/image' + str(num) + '.png', frame)
        num += 1
        count = 0
        print(num)