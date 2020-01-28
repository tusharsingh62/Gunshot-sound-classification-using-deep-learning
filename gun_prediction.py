from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input 
from keras.models import load_model
import numpy as np
import cv2
import pandas as pd

model = load_model("my_model.h5") # Load trained model

df = pd.read_csv('labels.csv')
df.head()

n = len(df)
gun = set(df['gun'])
n_class = len(gun)
class_to_num = dict(zip(gun, range(n_class)))
print("Possible gun are ", class_to_num.keys())

X = cv2.imread('./Train/Data/train/HPR_10.png') # Path of Spectrogram image for prediction
X = cv2.resize(X, (299,299))
X = np.expand_dims(X, axis=0) 

width = 299
def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

inception_features = get_features(InceptionV3, X)
xception_features = get_features(Xception, X)
features = np.concatenate([inception_features, xception_features], axis=-1)
output = model.predict(features)
cls = np.argmax(output)

gun_name = getKeysByValue(class_to_num, cls)
print("Predicted gun is ",gun_name)
