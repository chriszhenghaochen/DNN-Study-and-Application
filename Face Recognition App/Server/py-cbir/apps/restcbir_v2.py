from flask import Flask, request
from flask_restful import Api
import os
import uuid
import json

import os
import os.path as osp
import sys
import random
import string
import hashlib
import numpy as np

#import tornado.web
#from tornado.escape import json_encode
import base64

# import Image
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
import pdb

#rootDir = '/home/ubuntu/Documents'

openfacedir = '/media/data/frmwrks/openface'
modelDir = osp.join(openfacedir, 'models/dlib', "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(modelDir)
netDir = osp.join(openfacedir, 'models/openface', 'nn4.small2.v1.t7')
net = openface.TorchNeuralNet(netDir, imgDim=96, cuda=False)

upload_prefix = '../static/upload/'
SETNAME = 'lfw_raw'

app = Flask(__name__)
# db = SQLAlchemy(app)r
UPLOAD_FOLDER = upload_prefix
# ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)


def embed_dist(x1, x2):
     diff = x1 - x2
     dist = np.dot(diff.T, diff)
     return dist


class LocalMatcher(object):
    def __init__(self, setname):
        self.trinet_index = self.load_triN('../conf/{:s}_triN.pkl'.format(setname))
 

    def load_triN(self, pin):
        with open(pin) as fh:
           data = cPickle.load(fh)
        return data

        
    def match_triN(self, img_dst):
        """Put the aligned face and net generated vector here"""
        img = cv2.imread(img_dst)
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
            print rep
            phash = LSH_sift(rep)
            if phash in self.trinet_index:
                identities.append((self.trinet_index[phash]['imgpath'], osp.basename(self.trinet_index[phash]['imgpath']),  0))
                print('found in db!')
            for v in self.trinet_index:
                identities.append((v['imgpath'], osp.basename(v['imgpath']), embed_dist(np.array(v['vec']), rep)))
                    
        sorted_list = sorted(identities, key=lambda d: d[2])
        return sorted_list[:20]
                                   

    def search(self, dst_thum, debug=True):

        triN_list = self.match_triN(dst_thum) 
        return triN_list



index_alg = None
def get_global_vars():
    global index_alg
    index_alg = LocalMatcher(SETNAME)
    return index_alg


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #pdb.set_trace()    
        extension = '.jpg' #os.path.splitext(file.filename)[1]
        f_name = str(uuid.uuid4()) + extension
        #jpgfile = request.files['file']
        #jpgfile.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
        with open(os.path.join(app.config['UPLOAD_FOLDER'], f_name), 'wb') as fh:
            fh.write(request.values['file'].decode('base64'))
        ## image compare here and response top 5 image with link
        #pdb.set_trace()
        if f_name:
            rawname = os.path.join(app.config['UPLOAD_FOLDER'], f_name)

            index_alg = get_global_vars()
            imlist = index_alg.search(rawname)
        else:
            err_msg = 'No file is uploaded'
        return json.dumps([{'fullpath': fullpath, 'shortpath': shortpath, 'dist': dist} for fullpath, shortpath, dist in imlist])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=59988, debug=True)
