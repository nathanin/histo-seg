'''
Preprocessing for the whole slide image

includes separating tissue from background
and processable from non-processable tissue

use a white threshold to determine tissue / background
use an AlexNet to determine processable / non-processable tissue

essentially I want a function that takes in WSI
and spits out tilemap
    Make the function agnostic to the tilesize
    i.e. return a very rough map, that will be later reshaped
    to fit over tilemap
    The row:col ratio should be similar

    favor over-calling tissue area.
    favor a liberal definition of processable
        really exclude:
            fatty tissue
            tears
            edges
            folds
            marker ***
        Use a different sort of algorithm for each and then at the end
        just take the union.


changelog
---------
2017-04-18:
    Initial

'''



'''
Method

0. Grayscale threshold the entire low-resolution image
    - Areas in the range of 2k - 4k in each dimension, no RAM problem
1. Partition the low resolution area into 2000 x 2000 fixed
2. For each tile,
    If the white are is < threshold,
        - Apply some metods to partition out a rough tumor mask

3. Output a boolean mask the same size as the low-res image

-- The output mask will be heavily resized. hopefully some information
about the body of tissue and processable area remains
. ON average, the downsampling will be ~10-fold.

-Nathan
'''


import cv2
import numpy as np
import os
import shutil


def extract_low_res(wsi, **kwargs):
    # Parse some arguments -- really just for practice
    img = wsi.read_region(
        (0,0),
        wsi.level_count-1,
        wsi.level_dimensions[-1]
    )
    img = np.asarray(img)
    img = img[:,:,:-1]  # Flip BGR to RGB ---???
    return img


def white_mask(img, **kwargs):
    # Parse kwargs -- really for practice it's not necessary
    img = cv2.cvtColor(img, cv2.RGB2GRAY)
    white = img < white_index

    # Do some smoothing?
    # white is a np.bool ndarray
    return white


def main():



    pass


if __name__ == '__main__':
    main()




