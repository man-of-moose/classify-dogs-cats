import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import zipfile
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import urllib
from PIL import Image
import requests


app = Flask(__name__)
model_file = "pickles/model.pickle"
model = pickle.load(open(model_file, 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # retrieve user input
    picture_url = request.form.get('image')

    # load and save image as jpg in satatic folder for serving
    im = Image.open(requests.get(picture_url, stream=True).raw)
    im.save("static/url_image.jpg")

    # load and convert image into array for model
    url_req = urllib.request.urlopen(picture_url, timeout=10)
    arr = np.asarray(bytearray(url_req.read()), dtype=np.uint8)
    img_array = cv2.imdecode(arr, 0)  # 'Load it as it is'
    img_array = cv2.resize(img_array, (100, 100))
    img_array = img_array/255
    img_array = img_array.reshape(-1, 100, 100, 1)

    # prediction
    prediction = model.predict(img_array)

    if prediction < 0.5:
        output = round((1-prediction[0][0])*100,2)
        out_text = "I am {}% confident that this is a cat".format(output)
    else:
        output = round(prediction[0][0]*100,2)
        out_text = "I am {}% confident that this is a dog".format(output)

    return render_template('result.html', prediction_text=out_text)


if __name__ == "__main__":
    app.run(debug=True)
