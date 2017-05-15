#!/usr/bin/python

'''
Get the weights of each class:

f(class) = frequency(class) / < total image area, where class is present >
---> f(class) = frequency(class) / (image_count(class) * 256*256)
weight(class) = median of f(class)) / f(class)

Based on the paper [ insert paper name ]

Nathan Ing implemented circa December, 2016
ing.nathany@gmail.com


---
Update history
May 2017 - functionized for command line use

'''


import os
import glob
import cv2
import numpy as np
import sys


def main(imgdir, class_num):
    # TODO add check for input sanity
    # imgdir = argv[0]
    # class_num = argv[1]
    # imgdir = '/home/nathan/semantic-pca/data/seg_0.8.1024/train/mask'
    # class_num = 5
    class_num = int(class_num)

    term = os.path.join(imgdir, '*.png')
    imgs = glob.glob(term)

    counts = np.zeros(shape=(len(imgs), class_num), dtype=np.float)
    present = np.zeros(shape=(len(imgs), class_num), dtype=np.bool)

    freqs = np.zeros(shape=(1, class_num), dtype=np.float)

    np.random.shuffle(imgs)

    # Assume that, if there's more than 30k images, then we've got it
    for index, img in enumerate(imgs[:30000]):
        im = cv2.imread(img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if im.max() == 255:
            im /= 255
            im = np.round(im)

        ux = np.unique(im)
        if index % 200 == 0:
            print index,
            print img
        for u in ux:
            msk = im == u
            counts[index, u] = msk.sum()
            present[index, u] = True

        for u in range(class_num):
            img_present = (present[:, u]).sum() * 256 * 256
            class_total = (counts[:, u]).sum()

            div = class_total / float(img_present)
            freqs[0, u] = div

            if index % 200 == 0:
                print "f({}) = {}".format(u, div)

    med_freqs = np.median(freqs)

    weights = np.zeros(shape=freqs.shape, dtype=np.float)

    weightfile = os.path.join(imgdir, 'class_weights.txt')
    with open(weightfile, 'w') as f:
        for u in range(class_num):
            w = med_freqs / freqs[0, u]
            weights[0, u] = w
            towrite = 'Weight({}) = {}'.format(u, w)
            print towrite
            f.write(towrite + '\n')


if __name__ == '__main__':
    print 'Got args: ', sys.argv
    if len(sys.argv) > 3:
        print "get_class_weights.py takes exactly two arguments"
    else:
        main(*sys.argv[1:])