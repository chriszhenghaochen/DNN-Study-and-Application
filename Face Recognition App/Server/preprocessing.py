import os.path as osp
import cv2
import openface
import pickle
import os


openfacedir = '/media/data/frmwrks/openface'
modelDir = osp.join(openfacedir, 'models/dlib', "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(modelDir)
netDir = osp.join(openfacedir, 'models/openface', 'nn4.small2.v1.t7')
net = openface.TorchNeuralNet(netDir, imgDim=96, cuda=False)

dataset = ["Xiuying_Wang", "Hui_Cui","Zhenghao_Chen", "Zhiyong_Wang", "Barack_Obama", "Donald_Trump", "Jianlong_Zhou"]
result = []

for person in dataset:
    for root, dirs, files in os.walk("py-cbir/static/dataset/raw/" + person + "/"):
        for filename in files:
            if (filename.endswith(".jpg") or filename.endswith(".png")):
                img_src = os.path.join(root, filename)
                img = cv2.imread(img_src)
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
                    reps = net.forward(alignedFace)

                    vec = []

                    for rep in reps:
                        vec.append(rep.item())

                    path = "/static/dataset/raw/" + person + "/" + filename
                    result.append({"vec": vec, "imgpath": path})

f = file('py-cbir/conf/lfw_raw_triN.pkl','w')
pickle.dump(result, f)
f.close()