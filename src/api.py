"""
Flask-restful API script for
1) getting random cat and dog images from Flickr
2) running the classifier on images

This is used by the webpage to classify images

For the "Random" button to work, you need Flickr keys in the "flickr_keys.json" file.
The file be in the same directory as the api.py contents should look like this:
{
  "public":"cb31d5deb5ed0a124376bf417dc621d0",
  "secret":"acbb82b3e62d12cb"
}
Those are example keys - they are not valid.  Go to flickr to get them.

12/14/2017
John Kwong


"""

import os, json, io, sys

script_directory = os.path.dirname(os.path.realpath(__file__))
print(script_directory)
# sys.path.append(os.path.join(script_directory, 'common'))
sys.path.insert(0,os.path.join(script_directory, 'common'))
model_directory = os.path.join(script_directory, 'models')

import numpy as np
import requests
from PIL import Image
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

import CatOrDogConv
import flickr_api.flickr_api

app = Flask(__name__)
api = Api(app)

# load the model
print('Loading model')
# catdog = CatOrDogConv.CatOrDogConv(os.path.join(model_directory, 'cat_dog_model.h5'))
catdog = CatOrDogConv.CatOrDogConvVGG16(os.path.join(model_directory, 'bottleneck_fc_model.h5'))

# You need to run a prediction or run _make_predict_function to compile the predict function ahead of time
#  to make threadsafe
#
# prob = catdog.PredictFilename('cat.jpg')
# print(prob)
# need to
catdog.model._make_predict_function()


# read the json file containing the keys
flickr_keys = json.load(open(os.path.join(script_directory, 'flickr_keys.json')))
flickr = flickr_api.flickr_api.FlickrFetchImages(flickr_keys['public'], flickr_keys['secret'])


# The following is a solution for this error message:
# No 'Access-Control-Allow-Origin' header is present on the requested resource.
# Origin 'null' is therefore not allowed access. The response had HTTP status code 500
@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response


# API classes
class GetRandomImage(Resource):
    def get(self):
        # search_term = np.random.choice(['cat', 'dog'])
        IMAGE_SIZE = 'url_n'

        # search_term = np.random.choice(['cat portrait', 'dog portrait', 'cats', 'dogs', 'cat head', 'dog head'])
        search_term = np.random.choice(['cat portrait', 'dog portrait','cat head', 'dog head'])

        image = flickr.GetRandomImage(search_key = search_term, image_type = IMAGE_SIZE)
        image['search_term'] = search_term
        return image

class HelloWorld(Resource):
    # def __init__(self):
    def get(self):
        return {'hello': 'world'}
    def post(self):
        json_data = request.get_json(force=True)
        url = json_data['url']
        print(url)
        data = requests.get(url).content
        img = Image.open(io.BytesIO(data))
        print("size")
        print(np.array(img).shape)
        # got image
        print(img)
        prob = float(catdog.PredictImage(np.array(img))[0][0])
        # return {'prob':prob, 'url':url}
        return jsonify(prob=prob, url=url)


api.add_resource(HelloWorld, '/')
api.add_resource(GetRandomImage, '/random_image')

if __name__ == '__main__':
    # app.run(debug=False)
    app.run(host='0.0.0.0', debug=False)