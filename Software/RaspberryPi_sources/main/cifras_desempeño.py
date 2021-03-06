###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

'''
python3 cifras_desempeño.py --dataset dataset --detector face_detection_model \
--embeddings output/embeddings_v2.pickle \
--embedding-model dlib_face_recognition_resnet_model_v1.dat \
--confidence_dec 0.5 --confidence_rec 0.5 \
--shape-predictor shape_predictor_68_face_landmarks.dat \
--shape-pred shape_predictor_5_face_landmarks.dat

'''

import threading
import cv2
import argparse
import os
import pickle
import time
import sys
import dlib
import numpy as np
import imutils
from imutils import paths
from imutils.face_utils import FaceAligner
import matplotlib.pyplot as plt
from functions_v4 import get_faces, recognize, train
from FrameProcessing import FrameProcessing
from imutils.video import FPS
from sklearn.metrics import classification_report, plot_confusion_matrix
# PARAMETERS

# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-d", "--detector", required=True,
				help="path to OpenCV's deep learning face detector")
ap.add_argument("-e", "--embeddings", required=True,
				help="path to serialized db of facial embeddings")
ap.add_argument("-m", "--embedding-model", required=True,
				help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c_d", "--confidence_dec", type=float, default=0.5,
				help="minimum probability to filter weak detections")
ap.add_argument("-c_r", "--confidence_rec", type=float, default=0.5,
				help="minimum probability to filter weak identificactions")
ap.add_argument("-p", "--shape-predictor", required=True,
				help="path to facial landmark predictor")
ap.add_argument("-b", "--shape-pred", required=True,
				help="path to facial land for vector")

args = vars(ap.parse_args())


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
							  "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
confianza_dec = args["confidence_dec"]
confianza_recon = args["confidence_rec"]

predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)
# load our serialized face embedding model from disk
print("[INFO] loading face model...")
embedder = dlib.face_recognition_model_v1(args["embedding_model"])
sp = dlib.shape_predictor(args["shape_pred"])
# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# grab the paths to the input images in our dataset
i = 0
for (rootDir, dirNames, filenames) in os.walk(args["dataset"]):
	if i == 0:
		imagePaths = dirNames
	i += 1

model, le = train(data)

calculos = {}
fps_count = FPS().start()
x_test = []
y_test = []
y_train = []
nombres = set(data["names"])
nombres.add('unknown')

for name in imagePaths:
	if name == 'emilia_clarke':
		stream = cv2.VideoCapture("videos_eff/{}.m4v".format(name))
	else:
		stream = cv2.VideoCapture("videos_eff/{}.avi".format(name))

	Coincidencias = 0
	Falsas_coincidencias = 0
	prob = []
	time.sleep(2.0)
	Caras = 0

	print('Persona a evaluar : {}'.format(name))
	while True:
		(grabbed, frame) = stream.read()

		# if the frame was not grabbed, then we have reached the
		# end of the stream
		if not grabbed:
			stream.release()
			time.sleep(2.0)
			break

		frame = imutils.resize(frame, width=450)
		detections = get_faces(detector, embedder, sp,frame, confianza_dec, fa)
		#[(face,vector,coordenada,imagen_completa)]
		face_data = [(*face, *recognize(face[1], model, le, confianza_recon)) for face in detections]
		#[(face,vector,coordenada,imagen_completa, nombre, prob)]
		for item in face_data:
			x_test.append(item[1])
			y_test.append(item[4])
			y_train.append(name)
			prob.append(item[5])
			if item[4] == name:
				Coincidencias += 1
			elif item[4] != 'unknown':
				Falsas_coincidencias += 1
			Caras += 1
		fps_count.update()
	calculos[name] = (Coincidencias,Caras,Falsas_coincidencias, np.mean(prob))

print(calculos)

fps_count.stop()

print(classification_report(y_test, y_train,
                            target_names = nombres))

disp = plot_confusion_matrix(model, x_test, y_test,
                             display_labels = nombres,
                             cmap=plt.cm.Blues,
                             normalize='true')

disp.ax_.set_title("Normalized confusion matrix")
disp.ax_.set_xticklabels(disp.ax_.get_xticklabels(), rotation=45,
horizontalalignment='right',fontsize='x-small')
print("Normalized confusion matrix")
print(disp.confusion_matrix)

plt.show()

print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
time.sleep(1)
