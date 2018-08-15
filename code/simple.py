import os
import argparse
import time
import random
from eval import *
from miscc.config import cfg, cfg_from_file
import warnings
warnings.filterwarnings("ignore")

def parse_args():
 parser = argparse.ArgumentParser(description='Train a AttnGAN network')
 parser.add_argument('--cfg', dest='cfg_file',
                     help='optional config file',
                     default='cfg/bird_attn2.yml', type=str)
 parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
 parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
 parser.add_argument('--manualSeed', type=int, help='manual seed')
 args = parser.parse_args()
 return args

if __name__ == '__main__':
  print('Running Simple Inference')
  # gpu based
  args = parse_args()
  cfg_from_file(args.cfg_file)
  cfg.CUDA = True
  # load word dictionaries
  wordtoix, ixtoword = word_index()
  # load models
  print('Loading Model...')
  text_encoder, netG = models(len(wordtoix))
  print('Models Loaded')
  seed = 100
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cfg.CUDA:
    torch.cuda.manual_seed_all(seed)
  caption = 'a green field with trees and mountain in the back'
  im = generate(caption, wordtoix, ixtoword, text_encoder, netG, False)
  name = 'RESULT.png'
  im.save(name, format="png")
  print('Done!')
