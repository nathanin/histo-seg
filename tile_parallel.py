#!/usr/bin/python

import sys

sys.path.insert(0, '/home/nathan/histo-seg/code')
import seg_pipeline

if __name__ == '__main__':
    # TODO fix these hard-coded paths to be passed in args
    writeto = '/home/nathan/histo-seg/pca/testers'
    sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4']

    weights = '/home/nathan/semantic-pca/weights/seg_0.6/norm_iter_150000.caffemodel'
    model_template = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'

    # For multiscale, these aren't needed.
    remove = True
    overlap = 64
    tilesize = 512
    writesize = 256

    filename = sys.argv[1]

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
        tileonly=True)
