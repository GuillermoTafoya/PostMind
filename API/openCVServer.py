import json
import base64
from deepface import DeepFace
import cv2
from gtts import gTTS
from hashlib import new
from flask import Flask
from flask import request, jsonify
import base64

app = Flask(__name__)
data = {}

@app.route('/api/image',methods=['PUT'])
def changeImage():
    try:
        img = request.json["img"]
        #Codigo q decodifica imagen
        with open("imageRecieved.jpg", "wb") as fh:
            img2decode = img.encode('utf-8')
            fh.write(base64.decodebytes(img2decode))
        return jsonify( {"state": "success"} )
    except (IOError, TypeError) as e:
        return jsonify({"error": e})

@app.route('/api/emotion',methods=['GET'])
def getEmotion():
    try:
        print("GET")
        analysis = DeepFace.analyze(img_path = "imageRecieved.jpg", actions = ["age", "emotion"])

        emotion = analysis["dominant_emotion"]
        return jsonify(emotion)
    except (IOError, TypeError) as e:
        return jsonify({"error": e})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)