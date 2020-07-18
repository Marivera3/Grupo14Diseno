###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

'''
python3 main_pi_v5.py --detector face_detection_model \
--embeddings output/embeddings.pickle \
--embedding-model openface_nn4.small2.v1.t7 \
--confidence 0.5 --shape-predictor shape_predictor_68_face_landmarks.dat
'''

import json
import threading
import datetime as dt
import cv2
import argparse
import os
import hashlib
import io
import pickle
import codecs
import time
import sys
from pyimagesearch.centroidtracker_cp_v2 import CentroidTracker
from pyimagesearch.trackableobject_cp_v2 import TrackableObject
import dlib
import numpy as np
import imutils
import datetime
import mongoengine as me
from gpiozero import LED
from pymongo import MongoClient
from subprocess import Popen, PIPE
from PIL import Image
from imutils.face_utils import FaceAligner
from functions_v4 import extract_faces, get_embeddings, get_faces, recognize, draw_frame, show_frame, train
from VideoGet import VideoGet
from SaveVideo import SaveVideo
from Person2DBv4 import Person2DB
from User.User import PersonRasp
from RegMode import RegMode
from FrameProcessing import FrameProcessing
from imutils.video import FPS

# PARAMETERS

reg_led = LED(17)
D_PROB = 0.5 # Probability value for detection
R_PROB = 0.7 # Probability value for recognition
REG_NUM = 30 # Number of frames for registration

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
				help="path to OpenCV's deep learning face detector")
ap.add_argument("-e", "--embeddings", required=True,
				help="path to serialized db of facial embeddings")
ap.add_argument("-m", "--embedding-model", required=True,
				help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
				help="minimum probability to filter weak detections")
ap.add_argument("-p", "--shape-predictor", required=True,
				help="path to facial landmark predictor")
ap.add_argument("-b", "--shape-predv2", required=True,
				help="path to facil land for vector")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#detector = cv2.CascadeClassifier('../BS/haarcascade')

predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = dlib.face_recognition_model_v1(args["embedding_model"])
sp = dlib.shape_predictor(args["shape_predv2"])

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

recognizer, le = train(data)

## INICIO ##

time.sleep(1.0)

## Start processes

# Video capture
video_getter = VideoGet(src = 0, name='Video Getter')
print('[INFO] Starting VideoGet...')
video_getter.start()
time.sleep(2.0)

# Video recording
pathIn= './videorecordings/13/'
pathOut = 'video_v1.avi'
fps = 25
SV = SaveVideo(name='VideoWriter', vg=video_getter, pathOut=pathIn+pathOut, fps=fps, encode_quality=95)
print('[INFO] Starting saving Video...')
SV.start()

# Register mode
reg_mode = RegMode('192.168.0.25', 8099)
#reg_mode = RegMode('127.0.0.1', 8099)
reg_mode.start()
Register_counter = 0
Register_buffer = []


# Add person 2 DB
p2db = Person2DB()
p2db.start()


# maxDisappeared: Frame para que desaparezca el objeto trackeado
# max Distance: Distancia entre centrides máxima para que desaparezca objeto
#               (desplazamiento entre frames)
ct = CentroidTracker(maxDisappeared=5, maxDistance=200)
trackers = []
trackers_esp32 = []
trackableObjects = {}
out = 0
skipped_frames = 2
out_prev = 0
cpt=0;
fps_count = FPS().start()
while True:

    # Rutina de registro
    if reg_mode is not None:
        if reg_mode.Register_mode:
            # Retrieve frame
            frame = video_getter.frame.copy()
            frame = imutils.resize(frame, width=800)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            (H, W) = frame.shape[:2]
            rects = []

            reg_led.on()
            detections = extract_faces(detector, frame, D_PROB)
            if len(detections[1]) == 1:
                Register_buffer.append(detections)
                Register_counter += 1
            if Register_counter > REG_NUM:
                reg_led.off()
                print("Processing face data")
                p2db.queue.append([reg_mode.name, reg_mode.surname])
                for item in Register_buffer:
                    vec_actual = get_embeddings(item[0], item[1], item[2], embedder, sp)
                    if vec_actual is not None:
                        data['embeddings'].append(vec_actual)
                        data['names'].append(reg_mode.name + '_' + reg_mode.surname)
                Register_counter = 0
                Register_buffer = []
                reg_mode.reset()
                recognizer, le = train(data)
                with open("./output/embeddingsNew.pickle", "wb") as file:
                    pickle.dump(data, file)
            print(f'frame {Register_counter}')
            continue

    '''
    # Rutina de registro
    if True:
        if reg_mode.Register_mode:
            detections = extract_faces(detector, frame, D_PROB)
            fps_count.update()
            if len(detections[1]) == 1:
                Register_buffer.append(detections)
                Register_counter += 1
            if Register_counter > REG_NUM:
                for item in Register_buffer:
                    vec_actual = get_embeddings(item[0], item[1], item[2], embedder, fa)
                    if vec_actual is not None:
                        data['embeddings'].append(vec_actual.flatten())
                        data['names'].append('arca')
                Register_counter = 0
                Register_buffer = []
                reg_mode = False
                recognizer, le = train(data)
                cpt = 100
            print(f'frame {Register_counter}')
            continue
    '''

    # Retrieve frame
    frame = video_getter.frame.copy()
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (H, W) = frame.shape[:2]
    rects = []

    if cpt % skipped_frames == 0:
        recon = []
        fotos = []
        ps = []
        trackers = []
        devices = []

        detections = get_faces(detector, embedder, sp,frame, D_PROB, fa)
        #[(face,vector,coordenada,imagen_completa)]
        face_data = [(*face, *recognize(face[1], recognizer, le, R_PROB)) for face in detections]
        #[(face,vector,coordenada,imagen_completa, nombre, prob)]
        for item in face_data:
            #Listas con nombres de reconocidos
            recon.append(item[4])
            fotos.append(item[0])
            ps.append(item[5])
            [devices.append(1) for i in range(len(face_data))]
        for face in detections:
            (startX, startY, endX, endY) = face[2]
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)
            trackers.append(tracker)
            # loop over the trackers
            for item in face_data:
                print('Detectado: ',item[4])

    else:
        for tracker in trackers:
            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    objects, names, images, probabilities, devicess, blocks  = ct.update(rects,recon, fotos, ps, devices)
    # loop over the tracked objects
    for (objectID, centroid),(ID, name),(I,im),(D,prob),(F,dev), (G,block) in zip(objects.items(),
            names.items(), images.items(), probabilities.items(), devicess.items(), blocks.items()):
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid, name, im, prob, dev)
        else:
            x = [c[0] for c in to.centroids]
            direction = centroid[0] - np.mean(x)
            to.centroids.append(centroid)

            # Si es que salio
            # check to see if the object has been counted or not
            '''
            print(f'to.counted: {to.counted}')
            print(f'to.block_n: {to.block_n}')
            print(f'to.sent: {to.sent}')
            print(f'to.device: {to.device}')
            print(f'direction: {direction}')
            '''
            print(f'centroid: {centroid}')
            if not to.counted and direction > 30  and centroid[0] > W//2 and to.device == 1:
                to.inn = True
                to.counted = True
#           elif not to.counted and direction > 0  and centroid[1] > H - 250 and to.device == 0:
#               to.inn = True
#               to.counted = True
        if block:
            to.block_n = True
            to.name = name

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        #Coordinamos el paquete de envio
        ## Envio de mails
        if not to.sent and to.block_n and to.counted:
            paquete = [to.prob, to.pic, to.reconocido, to.out, to.inn]
            if to.reconocido :
                #enviar mail
                paquete.append(to.name)
            else:
                paquete.append('unknown {}'.format(objectID))
                # enviar mails
            ##Paquete a enviar
            to.sent = True
            print((paquete[0], paquete[2:]))
            paquete.append(datetime.datetime.now())
            p2db.queue.append(paquete)

    #fps_count.update()
    cpt += 1
    out_prev = out

    #exitbool = show_frame(frame)
#   if False:#cpt > 100:
    if cpt > 30:
         SV.stop()
         fps_count.stop()
         print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
         print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
         time.sleep(1)
         video_getter.stop()
         p2db.stop()
         # db_client.close()
         break
