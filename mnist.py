#!/usr/bin/env python
# coding: utf-8
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import sys
from keras.models import Sequential
model = Sequential()
model.add(Convolution2D(filters=32,
                       kernel_size=(3,3),
                       activation='relu',
                       input_shape=(64,64,3)
                       )


model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))





model.add(MaxPooling2D(pool_size=(2, 2)))




model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
epochs_step=int(sys.argv[1])
epochs=int(sys.argv[2])

history = model.fit(
        training_set,
        steps_per_epoch=epochs_step,
        epochs=epochs,
        validation_data=test_set,
        validation_steps=10

accuracy = history.history['accuracy'][-1]

print("accuracy " + str(accuracy))

f = open("accuracy.txt", "w")
f.write(str(accuracy))
f.close()


