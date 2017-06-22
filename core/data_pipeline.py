'''
This script is for creating datasets from labelled image-annotation pairs

The assumption is the labels and annotations are named the same, with
annotations in PNG and labels in JPG images, located in respective folders

Several data augmentation strategies are applied.
    - random sub-sampling
    - color augmentation towards fixed targets
    - 90 degrees rotation


See ReadMe.md for a minimal use case

----------------------------------------------------
Arguments:
Position args: [1] scale - integer - square length
               [2] multiplicity - integer - # of sub-samples
               [3] dataset name - string

----------------------------------------------------
Usage:
$ python ~/histo-seg/core/data_pipeline.py 512 10 dataset_01


ing.nathany@gmail.com
nathan.ing@cshs.org

'''


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


'''
Utility from days gone by. Some annotations were appended with '_mask'
This caused some problems. So i removed them.
Pretty sure this is no longer in use, but...
'''
def remove_masktxt(path):
    contents = glob.glob(os.path.join(path, '*.png'))
    for c in contents:
        base = os.path.basename(c)
        newbase = base.replace('_mask', '')

        newc = c.replace(base, newbase)
        os.rename(c, newc)
    #/end for
#/end remove_masktxt


'''
Another helper from the old days.
We used to require a text file like:
list.txt:
    1.jpg 1.png
    2.jpg 2.png

Now that we can use LMDB to feed caffe, this isn't needed

It's still nice to have
'''
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
            #/end if
        #/end for
    #/end with

    return listfile
#/end makelist

'''
For visualizing image-mask pairs

Burns colors, defined in a look up table into the image

This uses that listfile, list.txt described above.

I guess that's why I keep that function around.
'''
def impose_overlay(listfile, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    #/end if

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
        #/end if
        cv2.imwrite(os.path.join(dst, writename), img)
    #/end for
#/end impose_overlay

'''
I guess this could be inside data_rotate

I thought it might be useful to have outside since it's a
pretty commonthing to do.

eh.
'''
def rotate(img, rotation_matrix):
    img = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(img.shape[0:2]))
    return img
#/end rotate


'''
Reads images in img_dir and rotates them `iters` times by 90deg

I should probably infer the center for each image,
but assume all of them are uniformly square.

This function appends an 'r' to the filename each rotation.
'''
def data_rotate(img_dir, iters, ext='jpg', mode='3ch', writesize=256):
    center = (writesize / 2 - 1, writesize / 2 - 1)
    rotation_matrix = cv2.getRotationMatrix2D(
        center=center, angle=90, scale=1.0)

    img_list = sorted(glob.glob(os.path.join(img_dir, '*.' + ext)))
    for name in img_list:
        if mode == '3ch':
            img = cv2.imread(name)
        elif mode == '1ch':
            #img = cv2.imread(name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            img = cv2.imread(name)
            #img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
        #/end if

        for k in range(iters):
            name = name.replace('.' + ext, 'r.' + ext)
            #print name
            img = rotate(img, rotation_matrix)
            cv2.imwrite(filename=name, img=img)
        #/end for
    #/end for

    print '\tDone rotating images in {}'.format(img_dir)
#/end data_rotate



'''
Applies the color targets in `l_mean_range` and `l_std_range` to
the images in img_dir, saving copies each time.

the range and stds were determined through trial and error. Could easily
be a random delta from the image mean and std.
However, this sometimes results in unnatural looking results. These settings
were chosen because they're similar to plausible staining characteristics.

Should still switch it to use a random delta though.

The function appends a 'c' to the filenames.
'''
def data_coloration(img_dir, mode, ext):
    '''
    LOL
    '''
    # TODO replace with random  numbers generated from uniform distrib.
    l_mean_range = (144.048, 130.22, 135.5, 140.0)
    l_std_range = (40.23, 35.00, 35.00, 37.5)

    img_list = sorted(glob.glob(os.path.join(img_dir, '*.' + ext)))
    for idx, name in enumerate(img_list):
        for LMN, LSTD in zip(l_mean_range, l_std_range):
            target = np.array([[LMN, LSTD], [169.3, 9.01], [105.97, 6.67]])
            name_out = name.replace('.'+ext, 'c.'+ext)
            if mode == 'feat':
                img = cv2.imread(name)
                # img = data.coloration(img, LMN, LSTD)
                img = cnorm.normalize(img, target)
                cv2.imwrite(filename=name_out, img=img)
            elif mode == 'anno':
                img = cv2.imread(name)
                cv2.imwrite(filename=name_out, img=img)
            # end if
        # /end for
        if idx % 500 == 0:
            print '\tcolorizing {} of {}'.format(idx, len(img_list))
        # /end if
    #/end for

    print '\tDone color augmenting images in {}'.format(img_dir)
#/end data_coloration


'''
Returns the bounding box coordinates given height, width, and
a window size ("edge").

returns (x1, x2, y1, y2)

use like this:
bbox = random_crop(h,w,edge)
img_crop = img[bbox[0]: bbox[1], bbox[0]:bbox[1], ...]
'''
## Moved from data.py 6-22-17
def random_crop(h, w, edge):
    minx = 0
    miny = 0
    maxx = w - edge
    maxy = h - edge
    x = np.random.randint(minx, maxx)
    y = np.random.randint(miny, maxy)
    x2 = x + edge
    y2 = y + edge

    return [x, x2, y, y2]
#/end random_crop


'''
Extracts n SQUARE sub-images from the images in img_list

I should check that the requested edge length is legal for all the members
of img_list. eh.

We then resize them all to a constant size, given by writesize.
Yeah it's a little confusing. But it's a good way to show the net images of
varying resolution.

optionally, pass in a list of corrdinates to use for the bounding boxes
if no coordinates are given, then generate some and return them.
There's definitely a better way to do this by just stacking the mask onto the
image and doing this once, instead of twice. Same with actually every
other function in here.
I'll implement it next time.

This function appends an 's' to each new subimage name
'''
def sub_img(img_list, ext, mode='3ch', edge=512, writesize=256, n=8, coords=0):
    # In contrast to split, do a random crop n times

    # img_list = sorted(glob.glob(os.path.join(path, '*.'+ext)))
    example = cv2.imread(img_list[0])
    h, w = example.shape[0:2]

    # Decide to use previous coordinates or to keep track of them
    # Keep track of the randomly generated coordinates
    if coords == 0:
        gencoord = True
        coords = [0] * len(img_list)
    else:
        gencoord = False

    for index, (name, c) in enumerate(zip(img_list, coords)):
        img = cv2.imread(name)
        name = name.replace('.' + ext, '_{}.{}'.format(edge, ext))

        # print coord
        if gencoord:
            coordsout = np.zeros(shape=(n, 4), dtype=np.uint32)

        for i in range(n):
            if gencoord:
                x, x2, y, y2 = random_crop(h, w, edge=edge)
                coordsout[i, :] = [x, x2, y, y2]
            else:
                x, x2, y, y2 = c[i, :]
            # /end if

            name = name.replace('.' + ext, 's.' + ext)

            if mode == '3ch':
                subimg = img[x:x2, y:y2, :]
                subimg = cv2.resize(
                    subimg,
                    dsize=(writesize, writesize),
                    interpolation=cv2.INTER_LINEAR)
            elif mode == '1ch':
                subimg = img[x:x2, y:y2, :]
                subimg = cv2.resize(
                    subimg,
                    dsize=(writesize, writesize),
                    interpolation=cv2.INTER_NEAREST)
            # /end if

            # this is always going to write
            # linter kinda insists on indenting it
            cv2.imwrite(filename=name, img=subimg)

            if gencoord:
                coords[index] = coordsout
            # /end if
        # /end for

    return coords
# /end sub_img


def delete_list(imglist):
    print 'Removing {} files'.format(len(imglist))
    for img in imglist:
        os.remove(img)
# /end delete_list



'''
Define a set of transformations, to be applied sequentially, to images.
For each image, track it's annotation image and copy the relevant transformations.

This should work for any sort fo experiment where
- annotation images are contained in one dir
- similary named source images are contained in their own dir
- we want them to be multiplied

'''
def multiply_data(src, anno, scales = [512], multiplicity = [9]):

    print '\nAffirm that files in\n>{} \nand \n>{} \nare not originals.\n'.format(
        src, anno)
    choice = input('I have made copies. (1/no) ')

    if choice == 1:
        print 'Continuing'
    else:
        print 'non-1 response. exiting TODO: Make this nicer'
        return 0
    # /end if

    if len(scales) != len(multiplicity):
        print 'Warning: scales and multiplicity must match lengths'
        return 0
    # /end if

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
    # /end for


    # Now it's OK to remove the originals
    delete_list(srclist)
    delete_list(annolist)

    data_coloration(src, 'feat', 'jpg')
    data_coloration(anno, 'anno', 'png')

    data_rotate(src, 3, ext='jpg', mode='3ch')
    data_rotate(anno, 3, ext='png', mode='1ch')
# /end multiply_data

def make_segmentation_training(src, anno, root, scales, multiplicity):
    multiply_data(src, anno, scales, multiplicity)
    return makelist(src, anno, root)
# /end make_segmentation_training

if __name__ == "__main__":
    scales = [512]
    multiplicity = [10]
    dataset_root = sys.argv[1]

    #dataset_root = '/home/nathan/semantic-pca/data/seg_0.9'

    root = os.path.join(dataset_root, 'train')
    #root = '/home/nathan/semantic-pca/data/seg_0.8/train'
    src = os.path.join(root, 'jpg')
    anno = os.path.join(root, 'mask')
    listfile = make_segmentation_training(src, anno, root, scales, multiplicity)
    ## TODO add option for drawing overlays
    # impose_overlay(listfile, os.path.join(root, 'anno_cmap'))

    # Validation, do less.
    multiplicity = [3]
    root = os.path.join(dataset_root, 'val')
    #root = '/home/nathan/semantic-pca/data/seg_0.8/val'
    src = os.path.join(root, 'jpg')
    anno = os.path.join(root, 'mask')
    listfile = make_segmentation_training(src, anno, root, scales, multiplicity)
    ## TODO add option for drawing overlays
    # impose_overlay(listfile, os.path.join(root, 'anno_cmap'))
