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

def jet(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='jet') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    retval = np.zeros(shape = (N, 4))
    for k in range(N):
    	retval[k, :] = map_index_to_rgb_color(k)

    # Strip the alpha channel 
    retval = retval[:,:3]

    print cv2.convertScaleAbs(retval * 255)
    return cv2.convertScaleAbs(retval * 255)

def hsv(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    retval = np.zeros(shape = (N, 4))
    for k in range(N):
    	retval[k, :] = map_index_to_rgb_color(k)

    # Strip the alpha channel 
    retval = retval[:,:3]

    print cv2.convertScaleAbs(retval * 255)
    return cv2.convertScaleAbs(retval * 255)