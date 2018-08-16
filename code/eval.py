from __future__ import print_function

import os
import sys
import torch
import io
import time
import numpy as np
from PIL import Image
import torch.onnx
from datetime import datetime
from torch.autograd import Variable
from miscc.config import cfg
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from werkzeug.contrib.cache import SimpleCache
cache = SimpleCache()

def vectorize_caption(wordtoix, caption, copies=2):
    tokens = caption.split(' ')
    cap_v = []
    for t in tokens:
        t = t.strip().encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i,:] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)
    return captions.astype(int), cap_lens.astype(int)

def generate(caption, wordtoix, ixtoword, text_encoder, netG, blob_service, copies=2):
    # load word vector
    captions, cap_lens  = vectorize_caption(wordtoix, caption, copies)
    n_words = len(wordtoix)

    # only one to generate
    batch_size = captions.shape[0]

    nz = cfg.GAN.Z_DIM
    captions = Variable(torch.from_numpy(captions), volatile=True)
    if (len(captions) > 0):
        cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)

        if cfg.CUDA:
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()
            noise = noise.cuda()

        #######################################################
        # (1) Extract text embeddings
        #######################################################
        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)
            

        #######################################################
        # (2) Generate fake images
        #######################################################
        noise.data.normal_(0, 1)
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

        # for j in range(batch_size):
        #     for k in range(len(fake_imgs)):
        #         im = fake_imgs[k][j].data.cpu().numpy()
        #         im = (im + 1.0) * 127.5
        #         im = im.astype(np.uint8)
        #         im = np.transpose(im, (1, 2, 0))
        #         im = Image.fromarray(im)
        #         name = str(k)+'.png'
        #         im.save(name, format="png")

        im = fake_imgs[2][1].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        return im

def word_index():
    ixtoword = cache.get('ixtoword')
    wordtoix = cache.get('wordtoix')
    if ixtoword is None or wordtoix is None:
        x = pickle.load(open('../data/coco/captions.pickle', 'rb'))
        ixtoword = x[2]
        wordtoix = x[3]
        del x
        cache.set('ixtoword', ixtoword, timeout=60 * 60 * 24)
        cache.set('wordtoix', wordtoix, timeout=60 * 60 * 24)

    return wordtoix, ixtoword

def models(word_len):
    text_encoder = cache.get('text_encoder')
    if text_encoder is None:
        text_encoder = RNN_ENCODER(word_len, nhidden=256)
        state_dict = torch.load('../DAMSMencoders/coco/text_encoder100.pth', map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        text_encoder.cuda()
        text_encoder.eval()
        #cache.set('text_encoder', text_encoder, timeout=60 * 60 * 24)

    netG = cache.get('netG')
    if netG is None:
        netG = G_NET()
        state_dict = torch.load('../models/coco_AttnGAN2.pth', map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        if cfg.CUDA:
            netG.cuda()
        netG.eval()
        #cache.set('netG', netG, timeout=60 * 60 * 24)
    return text_encoder, netG

def eval(caption):
    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))
    # load blob service
    #blob_service = BlockBlobService(account_name='attgan', account_key=os.environ["BLOB_KEY"])

    t0 = time.time()
    urls = generate(caption, wordtoix, ixtoword, text_encoder, netG, False)
    t1 = time.time()

    response = {
        'small': urls[0],
        'medium': urls[1],
        'large': urls[2],
        'map1': urls[3],
        'map2': urls[4],
        'caption': caption,
        'elapsed': t1 - t0
    }

    return response
