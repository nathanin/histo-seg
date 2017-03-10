#!/usr/bin/python

import sys

sys.path.insert(0, '/home/nathan/histo-seg/code')
import seg_pipeline


if __name__ == '__main__':

	writeto = '/home/nathan/histo-seg/pca'
	sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3']

	weights = '/home/nathan/semantic-pca//weights/seg_0.3/norm_iter_50000.caffemodel'
	model_template = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'

	remove = True

	tilesize = 256

	filename = sys.argv[1]

	seg_pipeline.run_inference(filename = filename,
                               writeto = writeto,
                               sub_dirs = sub_dirs,
                               tilesize = tilesize,
                               weights = weights,
                               model_template = model_template,
                               remove_first = remove,
                               tileonly = True)