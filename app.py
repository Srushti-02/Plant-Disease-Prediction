from flask import Flask, render_template, request
import numpy as np
import keras.models
import re
import sys
import os
import base64
from skimage import io, transform
sys.path.append(os.path.abspath("./model"))
from load import *

global graph, model

model, graph = init()

app = Flask(__name__)

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

@app.route('/')
def index_view():
    return render_template('index.html')

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict', methods=['GET' , 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = io.imread('output.png')
    x = np.invert(x)
    x = transform.resize(x, (255, 255))
    x = x.reshape(1, 255, 255, 1)

    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        idx = np.argmax(out)

        response = class_names[idx]
        return render_template('result.html', name=class_names[idx], description=response)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
