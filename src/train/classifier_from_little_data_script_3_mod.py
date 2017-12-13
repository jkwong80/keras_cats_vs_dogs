'''

My modification of fcholett/classifier_from_little_data_script_3.py

got the vgg16_weights.h5 from here:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
which I found here:
https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

This is a big 500MB file.


Original docstring:

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


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input


data_root_path = os.environ['DATA_PATH']

data_path = os.path.join(data_root_path, 'kaggle_cat_dog')

train_data_dir = os.path.join(data_path, 'train')
test_data_dir = os.path.join(data_path, 'test1')
validation_data_dir = os.path.join(data_path, 'validation')
validation2_data_dir = os.path.join(data_path, 'validation2')


# path to the model weights files.
# doens't seem like this is used:
# weights_path = os.path.join(data_root_path, 'keras', 'models', 'vgg16_weights.h5')

top_model_weights_path = os.path.join(model_directory, 'bottleneck_fc_model_weights.h5')
# dimensions of our images.
img_width, img_height = 150, 150

# training parameters
nb_train_samples = 24000
nb_validation_samples = 1000
epochs = 5
batch_size = 16


#  This is outdated
# # build the VGG16 network
# model = applications.VGG16(weights='imagenet', include_top=False)
# print('Model loaded.')
#
# # build a classifier model to put on top of the convolutional model
# top_model = Sequential()
# top_model.add(Flatten(input_shape=model.output_shape[1:]))
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1, activation='sigmoid'))
#
# # note that it is necessary to start with a fully-trained
# # classifier, including the top classifier,
# # in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# # add the model on top of the convolutional base
# model.add(top_model)

# from discussion session
input_tensor = Input(shape=(150, 150, 3))
base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights(top_model_weights_path)
model = Model(input=base_model.input, output=top_model(base_model.output))



# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# # # smaller learning rate
# rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=0.0)
# model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
#  modified last argument to "validation_steps=nb_validation_samples // batch_size"
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# save models to the model_directory
model.save_weights(os.path.join(model_directory, 'cat_dog_fine_tune_weights.h5'))  # always save your weights after training or during training
model.save(os.path.join(model_directory, 'cat_dog_fine_tune_model.h5')) # save the model
