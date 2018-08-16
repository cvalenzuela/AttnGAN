import os
import json
import base64
import common
import argparse
import time
import random
import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from eval import *
from simple import parse_args
from miscc.config import cfg, cfg_from_file
import warnings
warnings.filterwarnings("ignore")

# Server settings
PORT = 3332
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

# Model settings
args = parse_args()
cfg_from_file(args.cfg_file)
cfg.CUDA = True
wordtoix, ixtoword = word_index()
print('Loading Model...')
text_encoder, netG = models(len(wordtoix))
print('Models Loaded')
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cfg.CUDA:
  torch.cuda.manual_seed_all(seed)

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
  caption = request["caption"]
  if (len(caption) > 3):
    img = generate(caption, wordtoix, ixtoword, text_encoder, netG, False)
    if img is not None:
      buff = io.BytesIO()
      img.save(buff, format="JPEG")
      string_img = base64.b64encode(buff.getvalue()).decode("utf-8")
      emit('update_response', {"image": string_img})

# Run the app
if __name__ == "__main__":
  print('Running AttnGAN Sever')
  socketio.run(app, host='0.0.0.0', port=PORT, debug=False)

