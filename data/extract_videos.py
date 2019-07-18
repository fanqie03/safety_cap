from mtcnn.mtcnn import MTCNN
import cv2
import argparse
from pathlib import Path
import os
import time
import random
import numpy as np
import nltk
from scipy import misc
from tools import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/home/mf/datasets/helmet')
    parser.add_argument('--output_dir', default='/home/mf/datasets/helmet_output')
    parser.add_argument('--image_size', default=112)
    parser.add_argument('--margin', default=600)
    parser.add_argument('--confidence', default=0.9)
    parser.add_argument('--rot_k', default=0, type=int)
    parser.add_argument('--helmet_expands', default=0.5)
    return parser.parse_args()


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    args = get_args()
    check_dir(args.output_dir)

    input_dir = Path(args.input_dir)
    videos = input_dir.rglob('*')
    mtcnn = MTCNN()

    for i, video in enumerate(videos):
        print(i)
        cap = cv2.VideoCapture(str(video))
        video_name = video.name
        output_path = os.path.join(args.output_dir, video_name)
        check_dir(output_path)

        flag, frame = cap.read()



        while flag:

            frame = np.rot90(frame, args.rot_k)

            # cv2.imshow('1', frame)

            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     break

            faces = mtcnn.detect_faces(frame)

            for face in faces:
                box = face['box']
                confidence = face['confidence']
                if confidence < args.confidence:
                    continue
                file_name = '{:.6f}-{:.4f}.jpg'.format(random.random(), confidence)
                output_file = os.path.join(output_path, file_name)

                # img_size = [args.img_size, args.img_size]
                img = crop_with_margin(frame, box, expands=args.helmet_expands, image_size=args.image_size)

                cv2.imwrite(output_file, img)

            flag, frame = cap.read()


        cap.release()
