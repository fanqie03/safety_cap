import cv2
import argparse
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--video_capture', default=0)
# parser.add_argument('--video_capture', default='rtsp://192.168.1.70:554/ch1/stream2')
parser.add_argument('--helmet_model', default='output/autosave_simple_net.h5')
parser.add_argument('--image_size', default=112)

args = parser.parse_args()

cap = cv2.VideoCapture(args.video_capture)

helmet_detector = HelmetDetector(args)
face_detector = FaceDetector(args)

while True:
    _, frame = cap.read()

    faces = face_detector(frame)
    helmet_detector(faces)

    for face in faces:
        x, y = face.bbox[:2]
        cv2.putText(frame, str(face.helmet_type), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)

    cv2.imshow('1', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


