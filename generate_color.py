#!/usr/bin/python

##################################################################
##################################################################
###
###   				Color Wheel 
### From:
### http://stackoverflow.com/questions/14720331/
### 		how-to-generate-random-colors-in-matplotlib 
###
##################################################################
##################################################################

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import cv2
import numpy as np


def generate(n, whiteidx = None, cmap = 'Set1'):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=n-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap) 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    retval = np.zeros(shape = (n, 4))
    for k in range(n):
    	retval[k, :] = map_index_to_rgb_color(k)

    # Strip the alpha channel 
    retval = retval[:,:3]

    colors_ = cv2.convertScaleAbs(retval * 255)
    if whiteidx is not None:
        colors_[whiteidx, :] = [255, 255, 255]
    print colors_
    return colors_
