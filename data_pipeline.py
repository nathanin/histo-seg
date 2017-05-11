#!/usr/bin/python
'''
This script is for creating datasets from labelled image-annotation pairs

The assumption is the labels and annotations are named the same, with
annotations in PNG and labels in JPG images, located in respective folders

Several data augmentation strategies are applied.

The working functions are supplied as part of `data.py`

ing.nathany@gmail.com

'''

import data
import glob
import os
import cv2
import numpy as np
import sys


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

    lut = np.zeros((256, ), dtype=np.uint8)
    lut[:5] = [240, 170, 115, 60, 10]
    # print lut
    f = open(listfile, 'r')
    for k, line in enumerate(f):
        # print line.replace('\n','')
        srcp, annop = line.split()

        src = cv2.imread(srcp)
        anno = cv2.imread(annop)
        # print anno

        anno = cv2.LUT(anno, lut)
        anno = cv2.applyColorMap(anno, cv2.COLORMAP_JET)

        img = np.add(src * 0.6, anno * 0.5)
        img = cv2.convertScaleAbs(img)

        writename = os.path.basename(srcp)
        if k % 500 == 0:
            print 'Overlay {}'.format(k)
        cv2.imwrite(os.path.join(dst, writename), img)


def make_segmentation_training(src, anno, root, scales, multiplicity):
    data.multiply_data(src, anno, scales, multiplicity)
    return makelist(src, anno, root)



'''

For use with a script that calls data_pipeline.py /path/to/new_dataset

'''
if __name__ == "__main__":
    scales = [726]
    multiplicity = [24]
    dataset_root = sys.argv[1]
    #dataset_root = '/home/nathan/semantic-pca/data/seg_0.9'

    root = os.path.join(dataset_root, 'train')
    #root = '/home/nathan/semantic-pca/data/seg_0.8/train'
    src = os.path.join(root, 'jpg')
    anno = os.path.join(root, 'mask')
    listfile = make_segmentation_training(src, anno, root, scales, multiplicity)
    impose_overlay(listfile, os.path.join(root, 'anno_cmap'))

    # Validation, do less.
    multiplicity = [2]
    root = os.path.join(dataset_root, 'val')
    #root = '/home/nathan/semantic-pca/data/seg_0.8/val'
    src = os.path.join(root, 'jpg')
    anno = os.path.join(root, 'mask')
    listfile = make_segmentation_training(src, anno, root, scales, multiplicity)
    impose_overlay(listfile, os.path.join(root, 'anno_cmap'))
