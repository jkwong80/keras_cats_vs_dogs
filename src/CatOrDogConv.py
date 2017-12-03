from keras.models import load_model

from PIL import Image
import os
import numpy as np

# dimensions of our images that it is reshaped to
# ---really hsould figure out how to get the input size of the network
img_width, img_height = 150, 150

class CatOrDogConv(object):
    """

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


