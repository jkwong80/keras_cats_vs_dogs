"""
Manually predict for images using the trained model.

11/30/2017, John Kwong


"""

import os, sys, glob

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_directory, '../common'))

import numpy as np
from keras.models import load_model
from PIL import Image
from keras import applications

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# dimensions of our images.
img_width, img_height = 150, 150

data_root_path = os.environ['DATA_PATH']

data_path = os.path.join(data_root_path, 'kaggle_cat_dog')

train_data_dir = os.path.join(data_path, 'train')
test_data_dir = os.path.join(data_path, 'test1')
validation_data_dir = os.path.join(data_path, 'validation')
validation2_data_dir = os.path.join(data_path, 'validation2')

class_name_list = ['cats', 'dogs']

validation_set_dir = {class_name:os.path.join(validation_data_dir, class_name) for class_name in class_name_list}
train_set_dir = {class_name:os.path.join(train_data_dir, class_name) for class_name in class_name_list}


# load the trained keras model
model = load_model(os.path.join(os.environ['HOME'], 'repo', 'keras_cats_vs_dogs', 'src', 'models', 'bottleneck_fc_model.h5'))
model_vgg16 = applications.VGG16(include_top=False, weights='imagenet')

# get list of files in
fullfilenamelist = {class_name:glob.glob(os.path.join(validation_set_dir[class_name], '*.jpg')) for class_name in class_name_list }
# fullfilenamelist = {class_name:glob.glob(os.path.join(train_set_dir[class_name], '*.jpg')) for class_name in class_name_list }

for class_name in class_name_list:
    fullfilenamelist[class_name].sort()

prediction_prob = {class_name:np.zeros(len(fullfilenamelist[class_name])) for class_name in class_name_list}

for class_name in ['cats']:
    print('Working on {}'.format(class_name))
    for instance_index, f in enumerate(fullfilenamelist[class_name]):
        if instance_index % 50 == 0:
            print('{}/{}'.format(instance_index, len(fullfilenamelist[class_name])))
        # img = Image.open(f).resize((img_width, img_height), Image.ANTIALIAS)
        # x = np.expand_dims(np.array(img), axis=0)
        # array_vgg16 = model_vgg16.predict(x)


        x = load_img(f, target_size=(img_width, img_height))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        array_vgg16 = model_vgg16.predict(x)

        prediction_prob[class_name][instance_index] = model.predict_proba(array_vgg16)


        if class_name == 'cats':
            if prediction_prob[class_name][instance_index] > 0.5:
                print('{}, {}, {}, wrong'.format(instance_index, f, prediction_prob[class_name][instance_index]))
        else:
            if prediction_prob[class_name][instance_index] < 0.5:
                print('{}, {}, {}, wrong'.format(instance_index, f, prediction_prob[class_name][instance_index]))
