from flask import Flask, render_template, url_for, request, redirect
import json
import numpy as np
from PIL import Image
import re
import base64
import sys
import os

sys.path.append(os.path.abspath('./model'))
from load import *
app = Flask(__name__)

global model, graph
model, graph = init()

global output 
output = 0
def convertImage(imagData):
    imgstr = re.search(r'base64,(.*)', str(imagData)).group(0)
    imgstr = imgstr.replace(" ","+")
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr[7:-3]))

@app.route('/', methods=['GET'])
def servePage():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def servePagePost():
    data = request.get_data()
    convertImage(data)
    image = Image.open('output.png').convert("L")
    image = image.resize((28,28))
    image_arr = np.array(image).reshape((1,28,28,1))
    out = model.predict_classes(image_arr/255)
    print(str(out))
    return str(out)


if __name__ == "__main__":
    app.run(debug=True)