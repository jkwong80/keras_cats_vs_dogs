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

print('loading model')
catdog = CatOrDogConv.CatOrDogConv(os.path.join(model_directory, 'cat_dog_model.h5'))
# catdog = CatOrDogConv.CatOrDogConv(os.path.join(model_directory, 'bottleneck_fc_model.h5'))

# prob = catdog.PredictFilename('cat.jpg')
# print(prob)
# need to compile the predict function ahead of time to make threadsafe
catdog.model._make_predict_function()

# read the json file containing the keys

flickr_keys = json.load(open(os.path.join(script_directory, 'flickr_keys.json')))
flickr = flickr_api.flickr_api.FlickrFetchImages(flickr_keys['public'], flickr_keys['secret'])


# solution on
@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

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