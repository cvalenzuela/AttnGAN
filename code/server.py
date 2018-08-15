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
from miscc.config import cfg, cfg_from_file
import warnings
warnings.filterwarnings("ignore")

# Server settings
PORT = 5678
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

def parse_args():
 parser = argparse.ArgumentParser(description='AttnGAN Server')
 parser.add_argument('--cfg', dest='cfg_file',
                     help='optional config file',
                     default='cfg/bird_attn2.yml', type=str)
 parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
 parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
 parser.add_argument('--manualSeed', type=int, help='manual seed')
 args = parser.parse_args()
 return args

def main(SOME_INPUT, OPTIONS):
  '''
  This could be the main function that queries the model
  and returns something based on SOME_INPUT with OPTIONS
  '''
  return true

# Base route, functions a simple testing 
@app.route('/')
def index():
  caption = 'a green field with trees and mountain in the back'
  generation = generate(caption, wordtoix, ixtoword, text_encoder, netG, False)
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
  print('Running AttnGAN Sever')
  socketio.run(app, host='0.0.0.0', port=PORT, debug=True)

