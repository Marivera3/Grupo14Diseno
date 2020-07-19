###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

'''
python3 cifras_desempeño_rapido.py --dataset dataset2 --detector face_detection_model \
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
from functions_v4 import get_faces, recognize, train
from FrameProcessing import FrameProcessing
from imutils.video import FPS
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
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
y_test2 = []
nombres = set(data["names"])
nombres.add('unknown')
skip_frames = 3
c = skip_frames
for name in imagePaths:
	if name == 'emilia_clarke':
		#stream = cv2.VideoCapture("videos_eff/{}.avi".format(name))
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
		if c != skip_frames:
			c += 1
			continue

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
			y_test2.append(item[4])
			y_train.append(name)
			prob.append(item[5])
			if item[4] == name:
				Coincidencias += 1
				x_test.append(item[1])
				y_test.append(item[4])
			elif item[4] != 'unknown':
				Falsas_coincidencias += 1
				x_test.append(item[1])
				y_test.append(item[4])
			Caras += 1
		fps_count.update()
		c = 0
	calculos[name] = (Coincidencias,Caras,Falsas_coincidencias, np.mean(prob))

print(calculos)

fps_count.stop()

y_test1 = le.transform(y_test)
for x in le.classes_:
	if x not in x_test:
		y_test2.append('unknown')
		y_train.append(x)


print(classification_report(y_test2, y_train,
							target_names = nombres))



nombres_reales = y_test2

nombres_teoricos = y_train

nombres = set(data['names'])
nombres.add('unknown')

contador_teorico = {x:0 for x in nombres}

for i in nombres:
	contador_teorico[i] = list(nombres_teoricos).count(i)

contador = {x:0 for x in nombres}
contador_malo = {x:{y:0 for y in nombres} for x in nombres}

for name_exp,name_teo in zip(nombres_reales,nombres_teoricos):
	if name_exp == name_teo:
		contador[name_teo] += 1
	else:
		contador_malo[name_teo][name_exp] += 1
norm_conf = []
for i in nombres:
	sum = 0
	for j in nombres:
		sum += contador_malo[i][j]
	sum += contador[i]
	if sum == 0 :
		norm_conf.append(0)
	else:
		norm_conf.append(float(contador[i])/float(sum))
conf = []
for i in nombres:
	tem_conf = []
	for j in nombres:

		if i==j:
			tem_conf.append(contador[i])
		else:
			tem_conf.append(contador_malo[i][j])
	conf.append(tem_conf)
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(conf), cmap=plt.cm.jet,
				interpolation='nearest')
width = len(nombres)
height = len(nombres)
for k,x in enumerate(nombres):
	ax.annotate(str(contador[x]), xy=(k, k),
				horizontalalignment='center',
				verticalalignment='center')
for k,x in enumerate(nombres):
	for kk,y in enumerate(nombres):
		if k==kk:
			continue
		ax.annotate(str(contador_malo[x][y]), xy=(kk, k),
					horizontalalignment='center',
					verticalalignment='center')
cb = fig.colorbar(res)
clases = nombres
plt.xticks(range(width), clases,rotation=45,
horizontalalignment='right',fontsize='x-small')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.yticks(range(height), clases)
plt.savefig('confusion_matrix.png', format='png')
plt.show()



print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
time.sleep(1)
