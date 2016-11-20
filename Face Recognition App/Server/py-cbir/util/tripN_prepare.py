import cPickle
import os.path as osp


def main():
    labelfile = '../conf/imvec/labels.csv'
    repfile = '../conf/imvec/reps.csv'
    imgroot = '/static/dataset/lfw_raw'
    outdict = {}
    with open(labelfile) as fhlab, open(repfile) as fhrep:
        for label, vect in zip(fhlab, fhrep):
            vectfloat =[float(i) for i in vect.split(',')]
            name, imgname = label.split('/')[3:]
            imgname = imgname.strip().replace('.png', '.jpg')
            outdict[osp.join(imgroot, name, imgname)] = vectfloat 
    with open('../conf/lfw_raw_triN.pkl', 'w') as fh:
        cPickle.dump(outdict, fh)


if __name__ == '__main__':
    main()
