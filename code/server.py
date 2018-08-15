import os
import json
import base64
import common
import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

PORT = 5678
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

def main(SOME_INPUT, OPTIONS):
  '''
  This could be the main function that queries the model
  and returns something based on SOME_INPUT with OPTIONS
  '''
  return true

# Base route, functions a simple testing 
@app.route('/')
def index():
  return jsonify(status="200", message='attnGan is running', query_route='/query')

# When a client socket connects
@socketio.on('connect', namespace='/query')
def new_connection():
  print('Client Connect')
  emit('successful_connection', {"data": "connection established"})

# When a client socket disconnects
@socketio.on('disconnect', namespace='/query')
def disconnect():
  print('Client Disconnect')

# When a client sends data. This should call the main() function
@socketio.on('update_request', namespace='/query')
def new_request(request):
  results = main()
  emit('update_response', {"results": results})

# Run the app
if __name__ == "__main__":
  socketio.run(app, host='0.0.0.0', port=PORT, debug=True)

# Take in base64 string and return PIL image
def stringToImage(base64_string):
  imgdata = base64.b64decode(base64_string)
  return Image.open(io.BytesIO(imgdata))

# Convert PIL Image to an RGB image(technically a numpy array) that's compatible with opencv
def toRGB(image):
  return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)