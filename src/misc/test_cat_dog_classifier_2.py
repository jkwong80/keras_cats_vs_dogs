"""
Manually predict for images using the trained model.  Display performance results.

This script expects that there the following directory with images
$DATA_PATH/kaggle_cat_dog/train/cats
$DATA_PATH/kaggle_cat_dog/train/dogs
$DATA_PATH/kaggle_cat_dog/validation/cats
$DATA_PATH/kaggle_cat_dog/validation/dogs

The "cats" ("dogs") subdirectories contain only images of cats (dogs).
This script (and others in this repo) can be easily modified to perform classification on other binary sets of images.


11/30/2017, John Kwong


"""

import os, sys, glob

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_directory, '../common'))

import numpy as np
from keras.models import load_model
from PIL import Image
from keras import applications
from keras.preprocessing.image import img_to_array, load_img

# dimensions of our images.
img_width, img_height = 150, 150

# root location of the datasets
data_root_path = os.environ['DATA_PATH']
# root location of the training an d
data_path = os.path.join(data_root_path, 'kaggle_cat_dog')

# location of the training ata
train_data_dir = os.path.join(data_path, 'train')
# location of the validation data
validation_data_dir = os.path.join(data_path, 'validation')

class_name_list = ['cats', 'dogs']

validation_set_dir = {class_name:os.path.join(validation_data_dir, class_name) for class_name in class_name_list}
train_set_dir = {class_name:os.path.join(train_data_dir, class_name) for class_name in class_name_list}


# load the trained keras model
model = load_model(os.path.join(os.environ['HOME'], 'repo', 'keras_cats_vs_dogs', 'src', 'models', 'bottleneck_fc_model.h5'))
# load the vgg16 model
model_vgg16 = applications.VGG16(include_top=False, weights='imagenet')

# get list of files in validation directory
fullfilenamelist = {class_name:glob.glob(os.path.join(validation_set_dir[class_name], '*.jpg')) for class_name in class_name_list }

# # get list of files in the training directory
# fullfilenamelist = {class_name:glob.glob(os.path.join(train_set_dir[class_name], '*.jpg')) for class_name in class_name_list }

# sort the file names
for class_name in class_name_list:
    fullfilenamelist[class_name].sort()

# create data structure to store probability values
prediction_prob = {class_name:np.zeros(len(fullfilenamelist[class_name])) for class_name in class_name_list}

# loop through the two classes
for class_name in class_name_list:
    print('Working on {}'.format(class_name))
    for instance_index, f in enumerate(fullfilenamelist[class_name]):
        if instance_index % 50 == 0:
            print('{}/{}'.format(instance_index, len(fullfilenamelist[class_name])))

        # load the image to the appropriate size, normalize it and run it through the VGG16 model
        x = load_img(f, target_size=(img_width, img_height))
        x = img_to_array(x)/255.
        x = np.expand_dims(x, axis=0)
        array_vgg16 = model_vgg16.predict(x)

        # predict with trained model.
        prediction_prob[class_name][instance_index] = model.predict_proba(array_vgg16)

        # print predictions that are incorrect
        if class_name == 'cats':
            if prediction_prob[class_name][instance_index] > 0.5:
                print('{}, {}, {}, wrong'.format(instance_index, f, prediction_prob[class_name][instance_index]))
        else:
            if prediction_prob[class_name][instance_index] < 0.5:
                print('{}, {}, {}, wrong'.format(instance_index, f, prediction_prob[class_name][instance_index]))

    # Print summary
    # define prediction as the class with the highest probability
    prediction = prediction_prob[class_name] > 0.5
    if class_name == 'cats':
        number_correct = (prediction == 0).sum()
    else:
        number_correct = (prediction == 1).sum()
    accuracy = float( number_correct ) / len(prediction)
    print('Accuracy = {}'.format(accuracy))
    print('#correct = {}'.format(number_correct))

