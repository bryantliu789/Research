from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

print("Starting ")
eeg_fs = 250
eeg_chans = ["C3", "Cz", "C4"]
eog_chans = ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
all_chans = eeg_chans + eog_chans
event_types = {0:"left", 1:"right"}
pickle_file = "data.pkl"
print (pickle_file) 

import sys
sys.setrecursionlimit(10000)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten

train_df = pickle.load(open(pickle_file, "rb"))
train_df.head(2)
print ("train len=", len(train_df))

train_df.drop(["event_type"], axis=1).iloc[0].apply(lambda x:x.shape)
X = train_df.drop(["event_type"], axis=1).apply(lambda x:np.stack(x, axis=-1), axis=1)

X = np.array(X.values.tolist())
X = X.reshape(list(X.shape)+[1])
Y = train_df["event_type"].values.astype(float)
print ("x=", X.shape, " y=", Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y)
print ("xtrain=", x_train.shape, " xtest=", x_test.shape)
print ("ytrain=", y_train.shape, " ytest=", y_test.shape)

cnn = keras.Sequential([Conv2D(32, 3, activation="relu", input_shape= (4000, 3, 1)),
                        Conv2D(64, 1, activation="relu"),
                        Flatten(),
                        Dense(256, activation="relu"),
                        Dropout(0.2),
                        Dense(128, activation="relu"),
                        Dense(1, activation="sigmoid")])

opt = Adam(learning_rate=0.001)
cnn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = cnn.fit(x_train, y_train, epochs=8, batch_size=64)

cnn.evaluate(x_test, y_test);

print("End ")