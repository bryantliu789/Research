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

test_file = "data.pkl"
print (test_file)

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

train_df = pickle.load(open(test_file, "rb"))
print ("train len=", len(train_df))
train_df.head(2)
print (train_df.head(2))

train_df.drop(["event_type"], axis=1).iloc[0].apply(lambda x:x.shape)

X = train_df.drop(["event_type"], axis=1).apply(lambda x:np.concatenate(x), axis=1)

X = np.array(X.values.tolist())
Y = train_df["event_type"].values.astype(float)
print ("x=", X.shape, " y=", Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y)
print ("xtrain=", x_train.shape, " ytrain=", x_test.shape)

neural_network = keras.Sequential([Dense(2048, activation="relu"),
                                   Dense(1024, activation="relu"),
                                   Dense(512, activation="relu"),
                                   Dense(128, activation="relu"),
                                   Dense(1, activation="sigmoid")])

opt = Adam(learning_rate=0.0001)
neural_network.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

history = neural_network.fit(x_train, y_train, epochs=15, batch_size=64)

fig, axs = plt.subplots(1, 2, figsize=(15, 4))
axs[0].plot(history.history["loss"])
axs[0].set_title("Neural Network Training Loss")
axs[0].set_ylabel("Categorial Crossentropy")
axs[0].set_xlabel("Epoch")
axs[1].plot(history.history["accuracy"])
axs[1].set_title("Neural Network Training Accuracy")
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("Epoch")

neural_network.evaluate(x_test, y_test)

print("End ")