# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 01:11:25 2021

@author: souha
"""

# Using K-Fold Cross Validation

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
X_test = sc.transform(X_test)

# Hyperparameter Optimization

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid

# For Grid Search

def create_model(layers, activation, optimizer):
    # Initialing the model
    model = Sequential()
    
    for i, nodes in enumerate(layers):
        if(i==0):
            model.add(Dense(nodes, input_dim=X_train.shape[1])) #For first input layer
        else: 
            model.add(Dense(nodes)) # For hidden layers
        # Common for input and hidden layers    
        model.add(Activation(activation))
        model.add(Dropout(0.3))
        
    # Output Layer
    model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
        
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# KerasClassifier is a wrapper of k-fold Cross Validation
model = KerasClassifier(build_fn=create_model, verbose=2)

# defining parameters for Grid Search
layers = [(20,), (40,20), (45,30,15),(6,6,1)]
activations = ['sigmoid', 'relu']
optimizers = [keras.optimizers.Adam(), keras.optimizers.RMSprop()]
# optimizers = ['adam', 'rmsprop']
param_grid = dict(
    layers=layers, 
    activation=activations, 
    optimizer = optimizers, 
    batch_size=[128, 256, 25, 32], 
    epochs=[30,100,500]
    )
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring = 'accuracy', cv=10)

grid_result = grid.fit(X_train, y_train)
[grid_result.best_score_,grid_result.best_params_]



# Using Cross Validation Score
# def build_classifier():
#     classifier = Sequential()
#     classifier.add(Dense(units=6, activation="relu", kernel_initializer="he_uniform", input_dim = 11))
#     classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
#     classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
#     classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#     return classifier    

# model = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs=100)
# cvs = cross_val_score(estimator= model, X=X_train, y=y_train, cv=10, verbose=2)

        