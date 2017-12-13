# Dog/Cat Convolution Neural Network Classifier

## Summary

This is a package for training a Convolutional Neural Network (CNN) for classifying images of dogs and cats with Keras (https://keras.io/) demonstrating it
with a simple website [www.dogorcat.online](http://www.dogorcat.online).  The model training files are modifications of those created by [François Chollet](https://github.com/fchollet) 
and can be found in this [blog post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).  (His scripts are unfortunately out-of-dat as Keras has chagned.)



## What is a Convolutional Neural Network

![convnet](src/web/images/convnet.png "Logo Title Text 2")

The Convolutional Neural Network is a particular type of neural network inspired by biological visual systems and have been proven to be very effective for image recognition.
The word "convolution" comes from the convolution layer which  consists of a dot product of a square grid of pixels ("filter").  The grid is translated across the image to form a feature map.  The application of various filters (defined during training) allow for the detection of features such as edges.  Please see the following resources for more information:

* [Neural networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
* [A great tutorial on CNNs](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
* [cs231n - Stanford class on CNN](http://cs231n.stanford.edu)


## Repository

This reposition contains the scripts for training the models, deploying the APIs and a basic website for demonstrating the trained model.  Below a description of the important files:

	- src
		- common
		   - flickr_api - class for interfacing with Flickr API
		- misc
		    - test_cat_dog_classifier_2.py - script for testing the "bottleneck" trained models
		- models
		    - bottleneck_fc_model.h5 - the model (keras)
		    - bottleneck_fc_model_weights.h5 - the model weights (keras)
		- train
		    - classifier_from_little_data_script_mod.py - training a Covnet from scratch.
		    - classifier_from_little_data_script_2_mod.py - training on "bottleneck" features calculated with trained VGG16 model
		- web
		- api.py - the flask API
		- start_api.sh - start the API
		- requirements.txt - list of python packages needed; use with pip
		- flickr_keys.json


The `classifier_from_little_data_script_mod.py` and `classifier_from_little_data_script_2_mod.py` are based on the [first](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d) and [second](https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069) scripts described 
in his [blog post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).  
I had to modify the code to accommodate changes in the new Keras 2.0 release and to work with where I like to place the data on my system (which will be described below).

## Setup

### Installing Packages
It is assumed that you are on a *nix system.  I developed this repo on my Macbook Pro with  MacOS 10.13.1 (High Sierra) and run the actually training on a Ubuntu 16.04 system.
Everything is written in Python 2.7.  The requirements.txt contains a list of the python packages needed which can be installed via pip.
>>pip install requirements.txt
I highly recommend training only with a GPU unless you want to wait forever.  I used a GTX 1080 
To use a GPU you need to install CUDA which you can obtain from Nvidia's website.

It is possible to attach an external GPU a Macbook Pro via thunderbolt port but this requires some serious mucking around. See [this](https://gist.github.com/jganzabal/8e59e3b0f59642dd0b5f2e4de03c7687). If you don't want to deal with hardware, I highly recommend using GPU instances on [AWS EC2](https://aws.amazon.com/ec2/instance-types/) or [Paperspace](https://www.paperspace.com/).

### Getting the Data
Download the data from [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).  You will need to create an account to access it.
The data will need to be organized as such:

    $DATA_PATH/kaggle_cat_dog/
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

You will need to create an environmental variable DATA_PATH ("export DATA_PATH=/Users/yourname/data")
Note that the dog and cat images are in separate directories.  The kaggle dataset as is does not separate the images.
Note also that you can also supplement with more data. I have yet to try this but here are some other public datasets of dogs and cats
* 


## Training
To train the model and save the models to file in the models src/models directory, run 
    python classifier_from_little_data_script_mod.py
or 
    python classifier_from_little_data_script_2_mod.py
Each script will save a _weights.h5 and a _model.h5 file.  The former contains on the weights (requiring you to build the model in code before loading) and the latter requires only the keras.models.load_model to instantiate the model.

## Website and API

To demonstrate the trained models, I developed a very rudimentary website [www.dogorcat.online](www.dogorcat.online) that runs the "bottleneck" classifier.  The user can run the classifier on a custom image by specifying the image URL before pressing the "Invoke" button.  Alternatively, the user can run the classifier on a random dog or cat image found on Flickr by pressing the "Random" button.

The APIs and the webpage are served by a low-end AWS Lightsail server.  The APIs are created with [Flask-RESTFul](https://flask-restful.readthedocs.io/en/latest/), which is a an extension for Flask for building REST APIs and is dead simple to use.  The webpage is served by Apache HTTP.

The API methods are documented in this Swaggerhub (OpenAPI) doc: [https://app.swaggerhub.com/apis/jkwong80/dogorcat/0.1.0]()https://app.swaggerhub.com/apis/jkwong80/dogorcat/0.1.0)

The webpage demonstrating