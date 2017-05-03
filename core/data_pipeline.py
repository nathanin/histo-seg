#!/usr/bin/python
'''
This script is for creating datasets from labelled image-annotation pairs

The assumption is the labels and annotations are named the same, with
annotations in PNG and labels in JPG images, located in respective folders

Several data augmentation strategies are applied.

The working functions are in the file `data.py`

ing.nathany@gmail.com
nathan.ing@cshs.org

'''

#import data
from openslide import OpenSlide
import cv2
import colorNormalization as cnorm
import numpy as np

import glob
import shutil
import os
import sys

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


def data_rotate(t, iters, ext='jpg', mode='3ch', writesize=256):
    center = (writesize / 2 - 1, writesize / 2 - 1)
    rotation_matrix = cv2.getRotationMatrix2D(
        center=center, angle=90, scale=1.0)

    img_list = sorted(glob.glob(os.path.join(t, '*.' + ext)))
    for name in img_list:
        if mode == '3ch':
            img = cv2.imread(name)
        elif mode == '1ch':
            #img = cv2.imread(name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            img = cv2.imread(name)
            #img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)

        for k in range(iters):
            name = name.replace('.' + ext, 'r.' + ext)
            #print name
            img = rotate(img, rotation_matrix)
            cv2.imwrite(filename=name, img=img)

    print '\tDone rotating images in {}'.format(t)



def data_coloration(t, mode, ext):
    '''
    LOL
    '''
    # TODO replace with random  numbers generated from uniform distrib.
    l_mean_range = (144.048, 130.22, 135.5, 140.0)
    l_std_range = (40.23, 35.00, 35.00, 37.5)

    img_list = sorted(glob.glob(os.path.join(t, '*.' + ext)))
    for idx, name in enumerate(img_list):
        if idx % 500 == 0:
            print '\tcolorizing {} of {}'.format(idx, len(img_list))
            for LMN, LSTD in zip(l_mean_range, l_std_range):
                name_out = name.replace('.' + ext, 'c.' + ext)
                if mode == 'feat':
                    img = cv2.imread(name)
                    img = coloration(img, LMN, LSTD)
                    cv2.imwrite(filename=name_out, img=img)
                elif mode == 'anno':
                    img = cv2.imread(name)
                    cv2.imwrite(filename=name_out, img=img)

    print '\tDone color augmenting images in {}'.format(t)



def delete_list(imglist):
    print 'Removing {} files'.format(len(imglist))
    for img in imglist:
        os.remove(img)

def multiply_data(src, anno, scales = [512], multiplicity = [9]):
    '''
    Define a set of transformations, to be applied sequentially, to images.
    For each image, track it's annotation image and copy the relevant transformations.

    This should work for any sort fo experiment where
    - annotation images are contained in one dir
    - similary named source images are contained in their own dir
    - we want them to be multiplied

    '''

    print '\nAffirm that files in\n>{} \nand \n>{} \nare not originals.\n'.format(
        src, anno)
    choice = input('I have made copies. (1/no) ')

    if choice == 1:
        print 'Continuing'
    else:
        print 'non-1 response. exiting TODO: Make this nicer'
        return 0

    if len(scales) != len(multiplicity):
        print 'Warning: scales and multiplicity must match lengths'
        return 0

    srclist = sorted(glob.glob(os.path.join(src, '*.jpg')))
    annolist = sorted(glob.glob(os.path.join(anno, '*.png')))

    # Multi-scale
    for scale, numbersub in zip(scales, multiplicity):
        print 'Extracting {} subregions of size {}'.format(numbersub, scale)
        coords = sub_img(
            srclist, ext='jpg', mode='3ch', edge=scale, n=numbersub)
        print 'Repeating for png'
        _ = sub_img(
            annolist,
            ext='png',
            mode='1ch',
            edge=scale,
            coords=coords,
            n=numbersub)


    # Now it's OK to remove the originals
    delete_list(srclist)
    delete_list(annolist)

    data_coloration(src, 'feat', 'jpg')
    data_coloration(anno, 'anno', 'png')

    data_rotate(src, 3, ext='jpg', mode='3ch')
    data_rotate(anno, 3, ext='png', mode='1ch')


def make_segmentation_training(src, anno, root, scales, multiplicity):
    multiply_data(src, anno, scales, multiplicity)
    return makelist(src, anno, root)


if __name__ == "__main__":
    scales = [1024]
    multiplicity = [35]
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
