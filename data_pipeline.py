#!/usr/bin/python

import data
import glob
import os
import cv2
import numpy as np
# /home/nathan/histo-seg/code/data_pipeline.py
# def make_classification_training(src):
#	 data.multiply_one_folder(src);

def remove_masktxt(path):
	contents = glob.glob(os.path.join(path, '*.png'))
	for c in contents:
		base = os.path.basename(c)
		newbase = base.replace('_mask', '')

		newc = c.replace(base, newbase)
		os.rename(c, newc)	


def makelist(src, anno, dst):
	print 'creating list'
	# list out the matching ones

	# Sometimes -- for some reason -- there won't be a match
	# Take the ones from src that match in anno
	listfile = os.path.join(dst, 'list.txt')

	srclist = sorted(glob.glob(os.path.join(src, '*.jpg')))
	srcbase = [os.path.basename(f).replace('.jpg', '') for f in srclist]

	annolist = sorted(glob.glob(os.path.join(anno, '*.png')))
	annobase = [os.path.basename(f).replace('.png', '') for f in annolist]

	with open(listfile, 'w') as f:
		for s, sb, a, ab in zip(srclist, srcbase, annolist, annobase):
			print '{} {}'.format(sb, ab)
			if sb == ab:
				f.write('{} {}\n'.format(s, a)) 

	return listfile

def impose_overlay(listfile, dst):
	if not os.path.exists(dst):
		os.mkdir(dst)

	lut = np.zeros((256,), dtype = np.uint8)
	lut[:4] = [135, 240, 78, 15]
	# print lut
	f = open(listfile, 'r')
	for line in f:
		# print line.replace('\n','')
		srcp, annop = line.split()		

		src = cv2.imread(srcp)
		anno = cv2.imread(annop)
		# print anno

		anno = cv2.LUT(anno, lut)
		anno = cv2.applyColorMap(anno, cv2.COLORMAP_JET)

		img = np.add(src*0.6, anno*0.5)
		img = cv2.convertScaleAbs(img)

		writename = os.path.basename(srcp)
		cv2.imwrite(os.path.join(dst, writename), img)



def make_segmentation_training(src, anno, root):
	remove_masktxt(anno)

	data.multiply_data(src, anno)

	return makelist(src, anno, root)



if __name__ == "__main__":

   # src = "/Users/nathaning/databases/pca/seg_0.3/feat"
   # anno = "/Users/nathaning/databases/pca/seg_0.3/anno_png"

   root = '/home/nathan/semantic-pca/data/seg_0.3'
   src = os.path.join(root, 'jpg_norm')
   anno = os.path.join(root, 'masks_png')

   listfile = make_segmentation_training(src, anno, root)

   impose_overlay(listfile, os.path.join(root, 'anno_cmap'))
