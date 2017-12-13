"""

These are classes that hold the covnet models trained with Keras by the first two methods in this blog post.
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


John Kwong


"""
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import applications

from PIL import Image
import os
import numpy as np

# dimensions of our images that it is reshaped to
# ---really hsould figure out how to get the input size of the network
img_width, img_height = 150, 150

class CatOrDogConv(object):
    """
        Cat or Dog classifier class.  The models are trained from scratch.
    """

    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(filename)
        self.model._make_predict_function()


        # need to get this from the model
        # width and height
        self.image_shape = [img_width, img_height]

    def PredictImage(self, img):
        """
        Predicts based on the image
        :param img:
        :return:
        """

        # turns the array into an image object, resizes it with antialiasing
        img = Image.fromarray(img).resize((self.image_shape[0], self.image_shape[1]), Image.ANTIALIAS)

        # convert to array, format to 4d and run prediction
        return(self.model.predict_proba(np.expand_dims(np.array(img), axis=0)))

    def PredictImageResized(self, img_resized):
        """
        Predicts based on the image that has already been resized
        :param img:
        :return:
        """

        return(self.model.predict_proba(np.expand_dims(np.array(img_resized), axis=0)))


    def PredictFilename(self, filename):
        """
        Loads the image from file and returns the probability classes
        :param filename:
        :return:
        """
        img_resized = Image.open(filename).resize((self.image_shape[0], self.image_shape[1]), Image.ANTIALIAS)
        return(self.PredictImageResized(img_resized))


class CatOrDogConvVGG16(object):
    """
        Cat or Dog classifier class.  It uses the VGG16 trained on ImageNet to generate the "bottleneck" features

        The classifier
    """

    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(filename)
        self.model._make_predict_function()
        self.model_vgg16 = applications.VGG16(include_top=False, weights='imagenet')


        # need to get this from the model
        # width and height
        self.image_shape = [img_width, img_height]

    def PredictImage(self, img):
        """
        Predicts based on the image
        :param img:
        :return:
        """

        # turns the array into an image object, resizes it with antialiasing
        img = Image.fromarray(img).resize((self.image_shape[0], self.image_shape[1]), Image.ANTIALIAS)

        x = np.expand_dims(np.array(img), axis=0)/255.
        array_vgg16 = self.model_vgg16.predict(x)

        # convert to array, format to 4d and run prediction

        return(self.model.predict_proba( array_vgg16 ))

    def PredictImageResized(self, img_resized):
        """
        Predicts based on the image that has already been resized
        :param img:
        :return:
        """
        x = np.expand_dims(np.array(img_resized), axis=0)/255.
        array_vgg16 = self.model_vgg16.predict(x)
        return(self.model.predict_proba( array_vgg16 ))


    def PredictFilename(self, filename):
        """
        Loads the image from file and returns the probability classes
        :param filename:
        :return:
        """

        x = Image.open(filename).resize((self.image_shape[0], self.image_shape[1]), Image.ANTIALIAS)

        return(self.PredictImageResized(x))


