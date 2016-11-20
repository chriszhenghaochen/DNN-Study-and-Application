import os
import os.path as osp
import sys
import random
import string
import hashlib
import numpy as np

import tornado.web
from tornado.escape import json_encode

import Image
import cv2

cur_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cur_path, '../util/')))
from img_hash import EXTS, phash, otsu_hash, otsu_hash2, hamming
from img_histo import gray_histo, rgb_histo, yuv_histo, hsv_histo, abs_dist
from img_gist import gist
from kmeans import eculidean_dist, norm0_dist
from img_hog import hog2, hog3, hog_lsh_list, hog_histo
from img_sift import sift2, sift_histo
from lsh import LSH_hog, LSH_sift
from rerank import blending, ensembling
import cPickle
import openface

rootDir = '/home/ubuntu/Documents'

modelDir = osp.join(rootDir, 'openface/models/dlib', "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(modelDir)
netDir = osp.join(rootDir, 'openface/models/openface', 'nn4.small2.v1.t7')
net = openface.TorchNeuralNet(netDir, imgDim=96, cuda=False)

upload_prefix = './static/upload/'
SETNAME = 'lfw_raw'
with open('static/url.pkl') as fh:
    urldat = cPickle.load(fh)

def embed_dist(x1, x2):
     diff = x1 - x2
     dist = np.dot(diff.T, diff)
     return dist


class LocalMatcher(object):
    def __init__(self, setname):
        self.trinet_index = self.load_triN('conf/{:s}_triN.pkl'.format(setname))
 

    def load_triN(self, pin):
        with open(pin) as fh:
           data = cPickle.load(fh)
        return data

        
    def match_triN(self, img_dst):
        """Put the aligned face and net generated vector here"""
        img = cv2.imread(img_dst)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        identities = []
        bb = align.getLargestFaceBoundingBox(img)
        bbs = [bb] if bb is not None else []
        for bb in bbs:
            landmarks = align.findLandmarks(img, bb)
            alignedFace = align.align(96, img, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE) 
            if alignedFace is None:
                continue
            rep = net.forward(alignedFace)
            phash = LSH_sift(rep)
            if phash in self.trinet_index:
                name = self.trinet_index[phash]['imgpath'].split('/')[-2].lower()
                     
                pname = self.trinet_index[phash]['imgpath'].split('/')[-2].replace('_', ' ')  
            #   identities.append((self.trinet_index[phash]['imgpath'], pname, 0.0, '0.0', urldat[name]))

                print('found in db!')
          
            for k, v in self.trinet_index.items():
                name = v['imgpath'].split('/')[-2].lower()
                pname = v['imgpath'].split('/')[-2].replace('_', ' ')
                identities.append((v['imgpath'], pname, embed_dist(np.array(v['vec']), rep), '{:.3f}'.format(embed_dist(np.array(v['vec']), rep)), urldat[name]))
                    
        sorted_list = sorted(identities, key=lambda d: d[2])
        return sorted_list[:20]
                                   

    def search(self, dst_thum, debug=True):

        return self.match_triN(dst_thum) 


index_alg = None
def get_global_vars():
    global index_alg
    index_alg = LocalMatcher(SETNAME)
    return index_alg


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('template_cbir.html', err_msg="", imgpath="", lists=[])

    def post(self):
        err_msg = ''
        img_path = ''
        lists = []
        debug_flag = self.get_argument("debug", "")
        if debug_flag:
            debug = True
        else:
            debug = False

        if self.request.files:
            f = self.request.files['imgpath'][0]
            rawname = f['filename']
            extension = os.path.splitext(rawname)[1]
            if extension[1:] not in EXTS:
                err_msg = 'wrong file type'
                self.render('template_cbir.html', err_msg=err_msg, imgpath=img_path, lists=[])

            #dstname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(10))
            dstname = hashlib.md5(f['body']).hexdigest()
            dstname += extension
            dst_full = upload_prefix + dstname
            with open(dst_full, 'w') as fh:
                fh.write(f['body'])
            #img = Imemage.open(dst_full)
            #img.thumbnail((50, 50), resample=1)
            #dst_thum = upload_prefix + 'thum_' + dstname 
            #img.save(dst_thum)

            index_alg = get_global_vars()
            lists = index_alg.search(dst_full)
        else:
            err_msg = 'No file is uploaded'
        self.render('template_cbir.html', err_msg=err_msg, imgpath=dst_full, lists=lists)

