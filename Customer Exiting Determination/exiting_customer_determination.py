# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:55:37 2021

@author: souha
"""

from keras.layers import LeakyReLU, PReLU, ELU
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Loading the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Loading categorical variables
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

# Concating categorical Variables
X = pd.concat([X, gender, geography], axis=1)

# Removing unnecessary data
X = X.drop(['Geography', 'Gender'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Building the ANN

# Initializing the ANN
classifier = Sequential()

# Adding first hidden layer and input layer
classifier.add(Dense(units=6, activation="relu",
               kernel_initializer="he_uniform", input_dim=11))
# For adding Dropout
# classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(
    Dense(units=6, kernel_initializer='he_uniform', activation='relu'))

# Adding the output layer
classifier.add(
    Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_history = classifier.fit(
    X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)


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

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting on custom value
# new_prediction = classifier.p

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
score = accuracy_score(y_pred, y_test)
