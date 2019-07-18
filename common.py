import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import cv2
import json

import tools

keras = tf.keras


class Face:
    def __init__(self):
        self.bbox = None
        self.container_image = None
        self.confidence = None
        self.raw_face = None

        self.helmet_type = None
        self.helmet_face = None


class FaceDetector:
    def __init__(self, args):
        self.args = args
        self.model = MTCNN()

    def __call__(self, frame, *args, **kwargs):
        rets = self.model.detect_faces(frame)
        faces = []
        for ret in rets:
            face = Face()
            face.bbox = ret['box']
            face.confidence = ret['confidence']
            face.container_image = frame
            faces.append(face)
        return faces


class HelmetDetector:
    def __init__(self, args):
        self.args = args
        with open(args.helmet_classes, 'r') as f:
            j = json.load(f)
            self.classes = dict(zip(j.values(), j.keys()))
        self.model = keras.models.load_model(args.helmet_model, compile=False)

    def __call__(self, faces, *args, **kwargs):
        for face in faces:
            face.helmet_face = img = tools.crop_with_margin(
                face.container_image, face.bbox, 0, self.args.image_size
            )
            # TODO bgr or rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            img = np.expand_dims(img, 0)
            ret = self.model.predict(img)
            index = np.argmax(ret)
            type = self.classes[index]

            face.helmet_type = type
