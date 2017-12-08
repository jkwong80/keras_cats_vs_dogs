'''

My modification of fcholett/classifier_from_little_data_script_2.py



Original docstring below

This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import os,sys
script_directory = os.path.dirname(os.path.realpath(__file__))
model_directory = os.path.join(script_directory, '../models')

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150


# define the directories, path to important files
top_model_weights_path = os.path.join(model_directory, 'bottleneck_fc_model_weights.h5')
top_model_path = os.path.join(model_directory, 'bottleneck_fc_model.h5')

data_root_path = os.environ['DATA_PATH']

data_path = os.path.join(data_root_path, 'kaggle_cat_dog')

train_data_dir = os.path.join(data_path, 'train')
test_data_dir = os.path.join(data_path, 'test1')
validation_data_dir = os.path.join(data_path, 'validation')
validation2_data_dir = os.path.join(data_path, 'validation2')


bottleneck_features_train_path = os.path.join(model_directory, 'bottleneck_features_train.npy')
bottleneck_features_validation_path = os.path.join(model_directory, 'bottleneck_features_validation.npy')

# training parameters
nb_train_samples = 24000
nb_validation_samples = 1000
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open(bottleneck_features_train_path, 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open(bottleneck_features_validation_path, 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open(bottleneck_features_train_path))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open(bottleneck_features_validation_path))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    # # smaller learning rate
    # rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=0.0)
    # model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)
    model.save(top_model_path)  # save the model


save_bottlebeck_features()
train_top_model()