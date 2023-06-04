import numpy as np
import cv2
import tensorflow as tf
from joblib import load
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from facenet_pytorch import MTCNN

# Model Architecture
def vgg_face():
    model = Sequential()
    # model layers definition omitted for brevity...
    model.add(Activation('softmax'))
    return model


# Prepare Training Data
train_dir='captured_frames'
Train_Data=tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rescale=1/255.0,
).flow_from_directory(train_dir,batch_size=16,target_size=(224,224),shuffle=False)


# Load VGG Face Model
model = vgg_face()
model.load_weights('vgg_face_weights.h5')
model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


# Extract Embedding Vectors
model = vgg_face()
model.load_weights('vgg_face_weights.h5')
model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
