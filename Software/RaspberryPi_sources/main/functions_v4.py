# USAGE

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.video import VideoStream
from imutils.video import FPS
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import argparse
import imutils
import pickle
import time
import dlib
import cv2
import os

def extract_faces(detector, frame, d_conf):
        # frame = imutils.resize(frame, width=500)
        # frame = vs.read()

        (h, w) = frame.shape[:2]
        size_array = np.array([w, h, w, h])

        imageBlob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = np.array(detector.forward())

        indexes = np.where(detections[0, 0, :, 2] > d_conf)[0]
        # This line gets all the index where detections has greater confidence than d_conf

        boxes = detections[0, 0, indexes, 3:7] * size_array
        lista = [frame, indexes, boxes]
        return lista

def get_embeddings(frame, indexes, boxes, embedder, shape_pred):
        coords = boxes.astype("int")
        i = indexes[0]
        (startX, startY, endX, endY) = coords[i]

        rect = dlib.rectangle(startX, startY, endX, endY)

        shape = shape_pred(frame, rect)
        #vec = np.asarray(embedder.compute_face_descriptor(frame, shape)).reshape(1, -1)
        vec = embedder.compute_face_descriptor(frame, shape)
        return vec

def get_faces(detector, embedder, shape_pred, frame, d_conf, facealigner):
        out_faces = []
        # frame = imutils.resize(frame, width=500)
        # frame = vs.read()

        (h, w) = frame.shape[:2]
        size_array = np.array([w, h, w, h])

        imageBlob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = np.array(detector.forward())

        indexes = np.where(detections[0, 0, :, 2] > d_conf)[0]
        # This line gets all the index where detections has greater confidence than d_conf

        boxes = detections[0, 0, indexes, 3:7] * size_array
        coords = boxes.astype("int")

        for i in indexes:
                (startX, startY, endX, endY) = coords[i]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                rect = dlib.rectangle(startX, startY, endX, endY)

                face = facealigner.align(frame, gray, rect)

                shape = shape_pred(frame, rect)
                vec = np.asarray(embedder.compute_face_descriptor(frame, shape)).reshape(1, -1)


                face = facealigner.align(frame, gray, rect)

                out_faces.append((face, vec, coords[i], frame))

        # frame, [(face, vector, coordinate),...]
        return out_faces



def recognize(vec, recognizer, le, r_conf=0.65):
        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        print(f"probabilidad: {proba}")
        if proba > r_conf:
                name = le.classes_[j]
        else:
                name = 'unknown'

        return name, proba

def acquire_frame(detector, embedder, frame, recognizer, le, d_conf, r_conf,faceA):
    out_faces = get_faces(detector, embedder, frame, d_conf,faceA)

    data = [(*face, *recognize(face[1], recognizer, le, r_conf)) for face in out_faces]
    return data

def draw_frame(frame, data):
        # draw the bounding box of the face along with the
        # associated probability
        coords = data[2]
        name = data[3]
        proba = data[4]
        (startX, startY, endX, endY) = coords
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame,
                                  (startX, startY),
                                  (endX, endY),
                                  (0, 0, 255), 2
        )
        cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return frame

def show_frame(frame):
        # show the output frame
        exitbool = False
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
                exitbool = True
        return exitbool


def train(data):
        # encode the labels
        print("[INFO] encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        return recognizer, le


def change_name(name,id_unknown, embeddings):
        i=0
        for name in embeddings['names']:
                if name==id_unknown:
                        embeddings['names'][i]=name
                i+=1
        print(dic)
