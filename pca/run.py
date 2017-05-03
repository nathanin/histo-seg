#!/usr/bin/python
import time
import sys
import os
import glob

'''
/home/nathan/histo-seg/pca/run.py

What we want is a set of images and their paired masks
The masks are label images range 0-n

n = background, so we have an ignore class

'''

# Add project code to python path:
sys.path.insert(0, '/home/nathan/histo-seg/code')
import seg_pipeline


def main(writeto, weights, model_template, filename):
    #writeto = '/home/nathan/histo-seg/pca/seg_0.8.1024_resume'
    sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4']

    #weights = '/home/nathan/semantic-pca/weights/seg_0.8.1024_resume/norm_iter_15000.caffemodel'
    #model_template = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'

    # Place holders;
    remove = False
    overlap = 64
    tilesize = 512
    writesize = 256

    # Set remove_first to FALSE
    # Run seg_pipeline:
    #for filename in filenames:
    seg_pipeline.run_multiscale(
        filename=filename,
        writeto=writeto,
        sub_dirs=sub_dirs,
        tilesize=tilesize,
        writesize=writesize,
        weights=weights,
        model_template=model_template,
        remove_first=remove,
        overlap=overlap,
        nclass=5,
        whiteidx=3,
        tileonly=False)


if __name__ == '__main__':
    writeto = sys.argv[1]
    weights = sys.argv[2]
    model_template = sys.argv[3]
    filename = sys.argv[4]

    print 'Writeto: {}'.format(writeto)
    print 'weights: {}'.format(weights)
    print 'model_template: {}'.format(model_template)
    print 'filename: {}'.format(filename)

    main(writeto, weights, model_template, filename)
