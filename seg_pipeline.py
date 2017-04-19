'''

Collection of run() functions from data, training and histoseg

'''
import os
import data
import histoseg  # Mine
import generate_color

import shutil
import glob
import inspect
import cv2

from openslide import OpenSlide
import numpy as np

import time

# Define inspection code that spits out the line it's called from (as str)
##################################################################
##################################################################
###
###       ~~~~~~~~ functions to do work ~~~~~~~~
###
##################################################################
##################################################################


# Inference for a single scale
# Records outputs to the Tiles Database
def run_inference(do_clean=True, do_parsing=True, **kwargs):
    start_time = time.time()
    if do_parsing:
        args = parse_options(**kwargs)
    else:
        args = kwargs

    exproot, expdirs, reportfile = data.make_inference(
        args['filename'], args['writeto'], args['sub_dirs'], args['tilesize'],
        args['writesize'], args['overlap'], args['remove_first'])

    repf = open(reportfile, 'a')

    # Check what mode to work in
    if args['tileonly']:
        print '\nDone processing {}; Returning\n'.format(args['filename'])
        end_time = time.time()
        elapsed = (end_time - start_time)
        print '\nTIME seg_pipeline.run_inference TILEONLY tilesize {} time: {}'.format(
            args['tilesize'], elapsed)

        #repf.write('TIME ALLTILES {}\n'.format(elapsed))
        repf.close()
        return

    # Main function here:
    repf.write('ENTERING INFERENCE\n')
    repf.close()
    histoseg.process(exproot, expdirs, args['model_template'],args['weights'],
                     args['caffe_mode'], args['GPU_ID'], reportfile)

    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME seg_pipeline.run_inference INFERENCE tilesize {} time: {}'.format(
        args['tilesize'], elapsed)

    return


##################################################################
##################################################################
###
###       ~~~~~~~~~~~~ Combine scales ~~~~~~~~~~~~
###
##################################################################
##################################################################

def run_multiscale(**kwargs):
    # scales = [556, 512, 496, 458]
    # 384 + 128 = 512 = native training resolution
    # 896 + 128 = 1024

    start_time = time.time()
    scales = [2100, 3000]
    #scales = [364, 384]

    for s in scales:
        # Re-parse, I guess
        args = parse_options(**kwargs)

        # Overwrite some settings
        args['tilesize'] = s  # Override tilesize
        args['sub_dirs'] = [
            '{}_{}'.format(subdir, args['tilesize'])
            for subdir in args['sub_dirs']
        ]
        run_inference(
            do_clean=False, do_parsing=False, **args)

    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME seg_pipeline.run_multiscale {}\n'.format(elapsed)
    return 0


##################################################################
##################################################################
###
###       ~~~~~~~~~~~~~~~ working functions ~~~~~~~~~~~~~~~~~
###
##################################################################
##################################################################


def parse_options(**kwargs):
    #print '\tParsing arguments: '
    defaults = {
        'filename': None,
        'writeto': None,
        'sub_dirs': ['tiles', 'result'],
        'tilesize': 512,
        'writesize': 256,
        'overlap': 32,
        'remove_first': False,
        'weights': None,
        'model_template': None,
        'caffe_mode': 0,
        'GPU_ID': 0,
        'overlay': True,
        'tileonly': False,
        'nclass': 5,
        'whiteidx': 0
    }

    # Check what is defined, and assign defaults:
    for d in defaults:
        if d in kwargs:
            pass
        else:
            #print '\t\tUsing default value for {}'.format(d)
            kwargs[d] = defaults[d]

    if None in kwargs.itervalues():
        raise Exception('All the paths must be set')

    return kwargs


