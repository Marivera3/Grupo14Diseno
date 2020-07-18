#!/bin/sh
python3 main.py --detector face_detection_model \
--embeddings output/embeddings_v2.pickle \
--embedding-model dlib_face_recognition_resnet_model_v1.dat \
--confidence 0.5 --shape-predictor shape_predictor_68_face_landmarks.dat \
--shape-predv2 shape_predictor_5_face_landmarks.dat
