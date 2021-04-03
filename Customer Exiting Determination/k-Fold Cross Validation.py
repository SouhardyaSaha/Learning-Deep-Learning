# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 01:19:11 2021

@author: souha
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:55:37 2021

@author: souha
"""

import pandas as pd

# Loading the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[: ,3:13]
y = dataset.iloc[:,13]

# Loading categorical variables
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

# Concating categorical Variables
X = pd.concat([X, gender, geography], axis=1)

# Removing unnecessary data
X=X.drop(['Geography', 'Gender'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train )
X_test = sc.fit_transform(X_test)


# Building the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding first hidden layer and input layer
classifier.add(Dense(units=6, activation="relu", kernel_initializer="he_uniform", input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs = 100)


import matplotlib.pyplot as plt
import numpy as np

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting on custom value
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])))
new_prediction = (new_prediction > 0.5)


# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="he_uniform", input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier    