import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pixel = pd.read_csv("digit-recognizer/train.csv").iloc[:, 1:].values
label = pd.read_csv("digit-recognizer/train.csv").iloc[:, 0].values


for i in range(len(pixel)):
    for j in range(len(pixel[i])):
        if pixel[i][j] < 170:
            pixel[i][j] = 0


plt.imshow(pixel[10].reshape(28, 28),cmap =plt.cm.gray_r, interpolation = "nearest")

x_train, x_test, y_train, y_test = train_test_split(pixel, label, test_size=0.2, random_state=4)

import keras
from keras.models import Sequential
from keras.layers import Dense
l = []

ann = Sequential()

ann.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=784))
ann.add(Dense(output_dim=6, init='uniform', activation='relu'))
ann.add(Dense(output_dim=6, init='uniform', activation='relu'))
ann.add(Dense(output_dim=10, init='uniform', activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(x_train, y_train, batch_size=30, nb_epoch=100)

yPred_ann = ann.predict(x_test)

for i in range(len(yPred_ann)):
    if yPred_ann[i] > 0.62:
        yPred_ann[i] = 1
    else:
        yPred_ann[i] = 0





