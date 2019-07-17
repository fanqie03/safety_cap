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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/home/mf/datasets/helmet')
    parser.add_argument('--output_dir', default='/home/mf/datasets/helmet_output')
    parser.add_argument('--img_size', default=112)
    parser.add_argument('--margin', default=600)
    return parser.parse_args()


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def crop_with_margin(img, det, margin, image_size):
    img_size = np.asarray(img.shape)[0:2]
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + det[0] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + det[1] + margin / 2, img_size[0])
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    return scaled


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

            frame = np.rot90(frame, 1)

            # cv2.imshow('1', frame)

            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     break

            faces = mtcnn.detect_faces(frame)

            for face in faces:
                box = face['box']
                confidence = face['confidence']
                file_name = '{:.4f}-{:.4f}.jpg'.format(random.random(), confidence)
                output_file = os.path.join(output_path, file_name)

                # img_size = [args.img_size, args.img_size]
                img = crop_with_margin(frame, box, margin=args.margin, image_size=args.img_size)

                cv2.imwrite(output_file, img)

            flag, frame = cap.read()


        cap.release()
