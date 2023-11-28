from flask import Flask, render_template, request
# from scipy.misc import imread, imresize
import imageio.v3 as iio
from PIL import Image
import numpy as np
from keras.models import model_from_json
import tensorflow as tf

json_file = open('model.json','r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("weights.h5")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

import re
import base64

# def convertImage(imgData1):
#     imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
#     with open('output.png', 'wb') as output:
#         output.write(base64.b64decode(imgstr))

def stringToImage(img):
    imgstr = re.search(r'base64,(.*)', str(img)).group(1)
    with open('image.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

# @tf.function
# def predict_in_graph_mode(input):
#     return model.predict(input)

@app.route('/predict/', methods=['POST'])
def predict():
    global model, graph
    imgData = request.get_data()
    try:
        stringToImage(imgData)
    except:
        f = request.files['img']
        f.save('image.png')

    # x = imread('image.png', mode='L')
    # x = imresize(x, (28, 28))
    # x = x.reshape(1, 28, 28, 1)

    x = iio.imread("image.png", mode = "L")
    x = np.array(Image.fromarray(x).resize((28, 28)))
    x = x.reshape(1, 28, 28, 1)

    # tf.compat.v1.disable_eager_execution()
    
    # with graph.as_default():
    #     prediction = model.predict(x)
    #     response = np.argmax(prediction, axis=1)
    
    tf.compat.v1.enable_eager_execution()   
    prediction = model.predict(x)
    response = np.argmax(prediction, axis=1)
    
    return str(response[0])


if __name__ == "__main__":

    # run the app locally on the given port
    app.run(host='0.0.0.0', port=80)
# optional if we want to run in debugging mode
    app.run(debug=True)
