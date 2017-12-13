# Dog/Cat Convolution Neural Network Classifier

## Summary

This is a package for training a Convolutional Neural Network (CNN) for classifying images of dogs and cats with [Keras](https://keras.io/) and demonstrating it with a simple website: [www.dogorcat.online](http://www.dogorcat.online).  The model training scripts are modifications of those created by [François Chollet](https://github.com/fchollet) to work with [Keras 2.0](https://blog.keras.io/introducing-keras-2.html) and with my preferred file structure.  The links to the original files can be found in this [blog post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).


## What is a Convolutional Neural Network

![convnet](src/web/images/convnet.png "Logo Title Text 2")

The Convolutional Neural Network is a particular type of neural network inspired by biological visual systems and have been proven to be very effective for image recognition.  The word "convolution" comes from the convolution layer which  consists of a dot product of a square grid of pixels ("filter").  The grid is translated across the image to form a feature map.  The application of various filters (defined during training) allow for the detection of features such as edges and even complicated shapes.  Please see the following resources for more information:

* [Neural networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
* [A great tutorial on CNNs](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
* [cs231n - Stanford class on CNN](http://cs231n.stanford.edu)


## Repository

This reposition contains the scripts for training the models, deploying the APIs and a basic website for demonstrating the trained model.  Below is a description of the important files:

	- src
		- train - scripts for trainig classifiers
		    - classifier_from_little_data_script_mod.py - training a small Covnet from scratch
		    - classifier_from_little_data_script_2_mod.py - training on "bottleneck" features 
		- models
		    - bottleneck_fc_model.h5 - the model (keras)
		    - bottleneck_fc_model_weights.h5 - the model weights (keras)
		- common
		   - flickr_api - class for interfacing with Flickr API
		- misc
			- test_cat_dog_classifier_2_mod.py - script for testing the "bottleneck" trained models calculated with trained VGG16 model
		- web - the website
		- api.py - the flask API
		- start_api.sh - start the API
		- requirements.txt - list of python packages needed; use with pip
		- flickr_keys.json - flickr public and secret keys


The `classifier_from_little_data_script_mod.py` and `classifier_from_little_data_script_2_mod.py` are based on the [first](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d) and [second](https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069) scripts described 
in his [blog post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).  I had to modify the code to accommodate changes in the new Keras 2.0 release and to work with my preferred location of data (which will be described below).

The flickr_keys.json file has two key-value pairs

	{
	  "public":"cb31d5deb5ed0a124376bf417dc621d0",
	  "secret":"acbb82b3e62d12cb"
	}
which you can easily obtain [here](https://www.flickr.com/services/apps/create/apply).  Don't try to use the above keys - they are fake.

## Setup

### Installing Packages
It is assumed that you are on a *nix system.  I developed this repo on my Macbook Pro with  MacOS 10.13.1 (High Sierra) and trained the models on a Ubuntu 16.04 system.
Everything is written in Python 2.7.  The requirements.txt contains a list of the python packages needed which can be installed via pip:

	pip install requirements.txt

I highly recommend training with a GPU unless you want to wait forever.  I used a GTX 1080 which trains the model in a reasonalbe amount of time.
To use a GPU, you need to install CUDA which you can obtain from Nvidia's website.  Some instructions on installing CUDA can be found [here](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04).

It is possible to attach an external GPU a Macbook Pro via thunderbolt port but this requires some serious mucking around. See [this](https://gist.github.com/jganzabal/8e59e3b0f59642dd0b5f2e4de03c7687). If you don't want to deal with hardware and installing all the requisite drivers and software, I highly recommend using GPU instances on [AWS EC2](https://aws.amazon.com/ec2/instance-types/) or [Paperspace](https://www.paperspace.com/).

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

You will need to create an environmental variable `DATA_PATH` (`"export DATA_PATH=/Users/yourname/data"`).  Note that the dog and cat images need to be in separate directories.  When you unzip the kaggle dataset file, you see that images have not been separated but this is simple enough to do.  Note also that you can supplement with more data.  I have yet to try this but here are some other public datasets of dogs and cats

* [CIFAR-10](https://www.kaggle.com/c/cifar-10/data)
* [STL-10](http://cs.stanford.edu/~acoates/stl10/)
* [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## Training
To train the model and save the models to file in the models src/models directory, run 

    python classifier_from_little_data_script_mod.py

or 

    python classifier_from_little_data_script_2_mod.py

The output should look something like this:

	Epoch 47/50
	24000/24000 [==============================] - 9s 365us/step - loss: 0.0652 - acc: 0.9815 - val_loss: 0.4223 - val_acc: 0.9270
	Epoch 48/50
	24000/24000 [==============================] - 9s 366us/step - loss: 0.0671 - acc: 0.9810 - val_loss: 0.4134 - val_acc: 0.9170
	Epoch 49/50
	24000/24000 [==============================] - 9s 365us/step - loss: 0.0620 - acc: 0.9821 - val_loss: 0.4156 - val_acc: 0.9190
	Epoch 50/50
	24000/24000 [==============================] - 9s 366us/step - loss: 0.0618 - acc: 0.9817 - val_loss: 0.4187 - val_acc: 0.9200


Each script will save a `_weights.h5` and a `_model.h5` file.  The former contains on the weights (requiring you to build the model in code before loading) and the latter requires only the keras.models.load_model to instantiate the model.  The first model achieves and accuracy of about 80%.  The second one does better at around 92%.

## Website and API

To demonstrate the trained models, I developed a very rudimentary website [www.dogorcat.online](http://www.dogorcat.online) that runs the "bottleneck" classifier.  The user can run the classifier on a custom image by specifying the image URL before pressing the "Invoke" button.  Alternatively, the user can run the classifier on a random dog or cat image found on Flickr by pressing the "Random" button.

The APIs and the webpage are served by a low-end AWS Lightsail server.  The APIs are created with [Flask-RESTFul](https://flask-restful.readthedocs.io/en/latest/), which is a an extension for Flask for building REST APIs and is dead simple to use.  The webpage is served by Apache HTTP.

The API methods are documented in this Swaggerhub (OpenAPI) doc: [https://app.swaggerhub.com/apis/jkwong80/dogorcat/0.1.0](https://app.swaggerhub.com/apis/jkwong80/dogorcat/0.1.0)
