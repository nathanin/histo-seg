#!/usr/bin/python

import numpy as np
import cv2
import sys
import os

sys.path.insert(0, '/home/nathan/caffe-segnet-cudnn5/python')
import caffe
sys.path.insert(0, '/home/nathan/histo-seg/code')
import histoseg
import generate_color


def run_validation(net, listfile, testingdir):
    '''
    For each line in `listfile`:
    split the line & pull the basename
    run the net forward once
    collect the result as an RGB class image
    impose colors
    write colors to testingdir/colored
    write labels to testingdir/labels
    '''
    # Check output paths:
    # colorwrite = os.path.join(testingdir, 'colored')
    # labelwrite = os.path.join(testingdir, 'labels')
    probwrite = os.path.join(testingdir, 'probabilities')
    # if not os.path.exists(colorwrite): os.mkdir(colorwrite)
    # if not os.path.exists(labelwrite): os.mkdir(labelwrite)
    if not os.path.exists(probwrite): os.mkdir(probwrite)

    # Instantiate the colors to use:
    colors = generate_color.generate(n=5, whiteidx=3, cmap='jet')
    i = 0
    with open(listfile, 'r') as f:
        for line in f:
            if i % 200 == 0:
                print 'image {}'.format(i)
            basename, _ = line.split()
            basename = os.path.basename(basename)
            basename,_ = os.path.splitext(basename)

            # colorname = os.path.join(colorwrite, '{}.jpg'.format(basename))
            # labelname = os.path.join(labelwrite, '{}.png'.format(basename))

            _ = net.forward()
            pred = net.blobs['prob'].data
            out = np.squeeze(pred)

            # color = histoseg.get_output('result', pred, out, colors)
            # label = np.argmax(out, axis=0)

            # cv2.imwrite(filename = colorname, img=colors)
            # cv2.imwrite(filename = labelname, img=label)

            for k in range(5):
                s = 'prob_{}'.format(k)
                sn = os.path.join(probwrite, '{}_{}.png'.format(basename, s))
                prob = histoseg.get_output(s, pred, out, colors, layer = k)
                cv2.imwrite(filename = sn, img = prob)

            i += 1








def main(dataname, listfile, modeldef, weights, writerep):

    testingdir = os.path.join(dataname, 'train')
    # modeldef = substitute_img_list(modeldef, testingdir, listfile, target='PLACEHOLDER')

    modeldef = '/home/nathan/semantic-pca/data/seg_0.8.1/val/test/segnet_basic_inference.prototxt'
    net = histoseg.init_net(modeldef, weights)
    run_validation(net, listfile, testingdir)
    del net







if __name__ == '__main__':
    dataname = '/home/nathan/semantic-pca/data/seg_0.8.1'
    listfile = os.path.join(dataname, 'train/list.txt')

    # Default to the segnet_basic in histo-seg/code
    modeldef = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'

    # Weights to use:
    weights = '/home/nathan/semantic-pca/weights/seg_0.8.1'
    weights = os.path.join(weights, 'norm_resumed_iter_32933.caffemodel')

    # Where to write the performance report
    writerep = os.path.join(dataname, 'performance.txt')
    main(dataname, listfile, modeldef, weights, writerep)
