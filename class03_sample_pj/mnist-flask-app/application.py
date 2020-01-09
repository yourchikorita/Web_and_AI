from flask import Flask, render_template, request
import numpy as np
import cv2
import json

from mnist_classifier import classify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/success/<int:number>")
def success(number):
    return render_template("success.html", res=number)


@app.route('/recognize', methods = ['POST'])
def upldfile():
    if request.method == "POST":
       file_val = request.data

       fig = cv2.imdecode(np.fromstring(request.data, np.uint8), cv2.IMREAD_UNCHANGED) #Convert from string to cv2 image
       img = cv2.cvtColor(fig, cv2.COLOR_RGB2GRAY) # Convert to grayscale

       prediction = classify(img)

       number = int(prediction)
       return \
           json.dumps({'num': number}), 200, {'ContentType': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
