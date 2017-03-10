#!/usr/bin/python

# Get the weights of each class:

# f(class) = frequency(class) / < total image area, where class is present >
# ---> f(class) = frequency(class) / (image_count(class) * 256*256)
# weight(class) = median of f(class)) / f(class)

import os
import glob
import cv2
import numpy as np

#imgdir = '/Users/nathaning/databases/ccRCC/for_segnet/mask_sub'
imgdir = '/home/nathan/semantic-pca/data/prepared_training/dec_6/mask'
class_num = 4 # range 0-3

term = os.path.join(imgdir, '*.png')
imgs = glob.glob(term)

counts = np.zeros(shape=(len(imgs), class_num), dtype=np.float)
present = np.zeros(shape=(len(imgs), class_num), dtype=np.bool)

freqs = np.zeros(shape=(1, class_num), dtype=np.float)

for index, img in enumerate(imgs):
	im = cv2.imread(img)
	ux = np.unique(im)
	if index % 200 == 0:
		print index,
		print img
	for u in ux:
		msk = im == u
		counts[index, u] = msk.sum()
		present[index, u] = True

		# New frequencies:
		# print "u: {} {}".format(u, msk.sum()),

	for u in range(class_num):
		img_present = (present[:,u]).sum()*256*256
		class_total = (counts[:,u]).sum()

		div = class_total / float(img_present)
		freqs[0,u] = div
		# print "{} / {} = ".format(class_total, img_present), 

		if index % 200 == 0:
			print "f({}) = {}".format(u, div)



med_freqs = np.median(freqs)

weights = np.zeros(shape=freqs.shape, dtype = np.float)

for u in range(class_num):
	w = med_freqs/freqs[0,u]
	weights[0,u] = w

	print "Weight({}) = {}".format(u, w)



