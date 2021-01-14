from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train = train.flow_from_directory('image_data/training/', target_size=(400, 200), batch_size=3, class_mode='binary')
validation = validation.flow_from_directory('image_data/validation/', target_size=(400, 200), batch_size=3, class_mode='binary')
test = test.flow_from_directory('image_data/testing/', target_size=(400, 200), batch_size=3, class_mode='binary')

print(train.class_indices)


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(400, 200, 3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation="relu"))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(train, epochs=10, validation_data=validation)
