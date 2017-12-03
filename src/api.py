from flask import Flask, jsonify, request
from flask_restful import Resource, Api

import numpy as np
import CatOrDogConv
import requests
import io
import requests
from PIL import Image

app = Flask(__name__)
api = Api(app)

print('loading model')
catdog = CatOrDogConv.CatOrDogConv('cat_dog_model.h5')

# prob = catdog.PredictFilename('cat.jpg')
# print(prob)
# need to compile the predict function ahead of time to make threadsafe
catdog.model._make_predict_function()

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

if __name__ == '__main__':
    app.run(debug=False)