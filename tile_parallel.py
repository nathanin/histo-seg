#!/usr/bin/python

import sys

sys.path.insert(0, '/home/nathan/histo-seg/code')
import seg_pipeline

if __name__ == '__main__':
    writeto = sys.argv[1]
    filename = sys.argv[2]

    #writeto = '/home/nathan/histo-seg/pca/seg_0.8.1024_resume'
    sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4']

    # For multiscale, these aren't needed.
    weights = 'dummy'
    model_template = 'dummy'
    remove = True
    overlap = 64
    tilesize = 512
    writesize = 256

    print 'Entering tile procedure...'
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
