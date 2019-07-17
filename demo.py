import cv2

cap = cv2.VideoCapture('rtsp://192.168.1.70:554/ch1/stream2')

while True:
    _, frame = cap.read()

    cv2.imshow('1', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break