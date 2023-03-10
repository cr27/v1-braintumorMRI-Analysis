from PIL import Image, ImageOps
import os, sys
import glob
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    MaxPooling2D,
)

yes_dir = "C:/Users/jbrow/OneDrive/Desktop/yesnotutor/yes"
no_dir = "C:/Users/jbrow/OneDrive/Desktop/yesnotutor/no"

yes_images = []
no_images = []

yes = []
no = []

for filename in glob.iglob(yes_dir + '**/*', recursive=True):
    #print(filename)
    im = Image.open(filename)
    im = ImageOps.grayscale(im)
    imResize = im.resize((128,128), Image.ANTIALIAS)
    imResize.save(filename , 'JPEG', quality=90)
    im = Image.open(filename)
    yes_images.append(im)

for filename in glob.iglob(no_dir + '**/*', recursive=True):
    #print(filename)
    im = Image.open(filename)
    im = ImageOps.grayscale(im)
    imResize = im.resize((128,128), Image.ANTIALIAS)
    imResize.save(filename , 'JPEG', quality=90)
    im = Image.open(filename)
    no_images.append(im)

for image in yes_images:
    I = np.asarray(image)
    yes.append(I)

for image in no_images:
    I = np.asarray(image)
    no.append(I)

X = yes+no
yes_label = np.ones(len(yes))
no_label = np.zeros(len(no))
y = np.concatenate((yes_label,no_label), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
X_train = np.asarray(X_train).reshape(189,128,128,1)
X_test = np.asarray(X_test).reshape(64,128,128,1)

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(128, 128, 1)),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25)
