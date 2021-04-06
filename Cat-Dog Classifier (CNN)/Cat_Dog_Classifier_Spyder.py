# Importing keras and packages
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

# Initializing the cnn
classifier = Sequential()

# step 1 Convolution
classifier.add(
    Conv2D(filters=32, kernel_size=3, input_shape=[
           64, 64, 3], activation='relu')
)
# Step 2 - Pooling
classifier.add(
    MaxPool2D(pool_size=2, strides=2)
)

# Adding 2nd convolution layer
classifier.add(
    Conv2D(filters=32, kernel_size=3, activation='relu')
)
classifier.add(
    MaxPool2D(pool_size=2, strides=2)
)

# Step 4 - Flattening our layer
classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Running the model
classifier.fit(
    training_set,
    epochs=25,
    validation_data=test_set,
)
#     steps_per_epoch=8000 // 32,
#     validation_steps = 2000 // 32

classifier.summary()
