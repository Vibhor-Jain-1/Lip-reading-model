import os
from keras.models import Sequential
from keras.layers import Activation, ZeroPadding3D, TimeDistributed, LSTM, GRU, Reshape
from keras.layers import Conv3D, MaxPooling3D,Dense, Dropout, Flatten
from keras import regularizers

import cv2
import dlib
import math
import json
import statistics
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import csv
from collections import deque
import tensorflow as tf
import sys
sys.path.append('D:\lip reading model\Computer-Vision-Lip-Reading-2.0-main\Computer-Vision-Lip-Reading-2.0-main')
from constants import *
from constants import TOTAL_FRAMES, VALID_WORD_THRESHOLD, NOT_TALKING_THRESHOLD, PAST_BUFFER_SIZE, LIP_WIDTH, LIP_HEIGHT


label_dict = {13:'you',6: 'hello', 5: 'dog', 10: 'my', 12: 'test', 9: 'lips', 3: 'cat', 11: 'read', 0: 'a', 4: 'demo', 7: 'here', 8: 'is', 1: 'bye', 2: 'can'}
count = 0
#label_dict = {2: 'my', 1: 'lips', 3: 'read', 0: 'demo'}

# Define the input shape
input_shape = (22, 80, 112, 3)

'''
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu'),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_dict), activation='softmax')
])
'''

model = Sequential()
model.add(Conv3D(8, (3, 3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Conv3D(32, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))


model.load_weights('D:\lip reading model\Computer-Vision-Lip-Reading-2.0-main\Computer-Vision-Lip-Reading-2.0-main\model/train_weights.weights.h5')

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("D:\lip reading model\Computer-Vision-Lip-Reading-2.0-main\Computer-Vision-Lip-Reading-2.0-main\model/face_weights.dat")

# read the image
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 60)
curr_word_frames = []
not_talking_counter = 0



first_word = True
labels = []

past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)