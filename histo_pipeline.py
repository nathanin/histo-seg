import data
import histoseg

import os
import glob
import time
import shutil

import cv2
import numpy as np
from openslide import OpenSlide


'''

histo_pipeline.py

ACTUAL pipeline not that fragmented thing called seg_pipeline
Do everything for each slide in a row, instead of separately.
That thing about splitting up different components of the pipeline
was a terrible idea
the worst. and i know a thing or two about bad ideas.

the worst part of that bad idea was all the options.
just get rid of all the options.


'''

def record_processing(repf, st):
    # Write string to reportfile
    f = open(repf, 'a')
    f.write(st)
    f.close()


def init_file_system(**kwargs):
    # Create the file system
    filename = kwargs['filename']
    tail = os.path.basename(filename)
    slide_name, ex = os.path.splitext(tail)
    exp_home = os.path.join(kwargs['writeto'], slide_name)

    s = 'Creating file system at : {}\n'.format(exp_home)
    record_processing(kwargs['reportfile'], s)

    # Sub-dirs hold tiles and the results
    if not os.path.exists(exp_home):
        try:
            os.makedirs(exp_home)
            for s in kwargs['scales']:
                d = os.path.join(exp_home, 'tiles_{}'.format(s))
                os.makedirs(d)
                for d in kwargs['outputs']:
                    d = os.path.join(exp_home, 'prob_{}_{}'.format(d,s)
                    os.makedirs(d)
                    s = 'Created {}\n'.format(d)
                    record_processing(kwargs['reportfile'], s)
        except:
            print 'Error initializing filesystem'
            print 'Attempting to create {}'.format(exp_home)
            print 'Attempting to create {}'.format(d)
            s = 'Failure intializing filesystem\n'.format(exp_home)
            record_processing(kwargs['reportfile'], s)
            return 0

    return exp_home


def whitespace(img, reportfile, white_pt=210):
    # Simple. Could be more sophisticated
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    white = gray > white_pt

    s = 'Found {} px above grayscale value {}\n'.format(
        white.sum(), white_pt
    )
    record_processing(reportfile, s)

    return white


def read_region(wsi, start, level, dims):
    # Utility function because openslide is weird
    img = wsi.read_region(
        start,
        level,
        dims
    )
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.cvtColor(img, cv2.BGR2RGB)

    return img


def get_process_map(img, tilesize, masks, reportfile):
    # For doing some tile-wise preprocessing; nothing else
    nrow, ncol = [img.shape[0] / tilesize,
                  img.shape[1] / tilesize]
    processmask = np.zeros(shape=img.shape[:2], dtype=np.bool)

    # Inverse the masks and take the union, by adding them together.
    inv = lambda x: 1-x
    masks = [inv(mask) for mask in masks]
    n_masks = len(masks)
    mask = np.add(masks)
    mask = mask == n_masks

    # Resize
    # The downsampling will be ~10x
    #mask = cv2.resize(mask, dsize=(nrow, ncol),
    #                  interpolation = cv2.INTER_NEAREST)

    # area
    mask_area = np.sqrt(mask.sum() / 2.5)
    s = 'Processable area is ~ {} sq. microns\n'.format(mask_area)
    record_processing(reportfile, s)

    return mask


def preprocessing(**kwargs):
    wsi = OpenSlide(kwargs['filename'])

    img = read_region(wsi,
                      (0,0),
                      wsi.level_count-1,
                      wsi.level_dimensions[-1]
                      )
    s = 'Preprocessing...\n'
    s = '{}Successfully read image from {}\n'.format(s,kwargs['filename'])
    record_processing(kwargs['reportfile'], s)

    # Boolean image of white areas
    whitemap = whitespace(img, kwargs['reportfile'])

    # TODO (nathan) add tumor location here
    #unprocessable = unprocessable_area(img)

    lo_res = 3000
    masks = [whitemap]
    process_map = get_process_map(img, lo_res, masks, kwargs['reportfile'])

    return process_map


def clean_intermediates(dirlist):
    # Bludgeon away the details
    pass


def tile_scale(**kwargs):
    # Perform tiling for one scale, according to the usual method.
    # Call data.tile_wsi
    writeto = 'tiles_{}'.format(scale)

    # Work with kwargs['processmap'] to get tilemap:
    wsi = kwargs['wsi']
    tilesize = kwargs['tilesize']
    if wsi.properties['aperio.AppMag'] == '20':
        level_20 = 0
    elif wsi.properties['aperio.AppMag'] == '40':
        level_20 = 1
    nrow, ncol = [wsi.level_dimensions[level_20][0] / tilesize,
                  wsi.level_dimensions[level_20][1] / tilesize]
    tilemap = cv2.reshape(kwargs['processmap'],
                          dsize=(nrow,ncol),
                          interpolation=cv2.INTER_NEAREST)
    s = 'Resized processmap from {} to {}\n'.format(
        kwargs['processmap'].shape, tilemap.shape
    )
    record_processing(kwargs['reportfile'], s)

    data.tile_wsi(
        wsi=kwargs['wsi'],
        tilesize=kwargs['tilesize'],
        writesize=kwargs['writesize'],
        writeto=writeto,
        overlap=kwargs['overlap'],
        prefix=kwargs['prefix'],
        tilemap=tilemap
    )


def process_scale(**kwargs):
    s = 'Processing scale {}\n'.format(kwargs['scale'])
    record_processing(kwargs['reportfile'])

    # Some hard coded constants in there.. that's OK for now
    tile_scale(wsi=kwargs['wsi'], tilesize=kwargs['scale'], writesize=256,
               overlap=64, prefix='tile', processmap=kwargs['processmap'],
               reportfile=kwargs['reportfile'])

    # Get the appropriate output dirs:
    expdirs = ['tiles_{}'.format(kwargs['scale'])]
    for output in kwargs['outputs']:
        expdirs.append('prob_{}_{}'.format(output, kwargs['scale']))

    # Call histoseg.process()
    s = 'Passing control to histoseg.process\n'
    record_processing(kwargs['reportfile'])
    histoseg.process(
        exphome=kwargs['exp_home'],
        expdirs=expdirs,
        model_tempplate=kwargs['model_template'],
        weights=kwargs['weights'],
        mode=1,
        GPU_ID=0,
        reportfile=kwargs['reportfile']
    )


def process_multiscale(**kwargs):
    s = 'Working on multiscale processing\n'
    record_processing(kwargs['reportfile'], s)

    wsi = OpenSlide(kwargs['filename'])

    start_time = time.time()
    for i,scale in enumerate(kwargs['scales']):
        process_scale(
            scale=scale,
            wsi=wsi,
            processmap=kwargs['processmap'],
            reportfile=kwargs['reportfile'],
            outputs=kwargs['outputs'],
            exp_home=kwargs['exp_home'],
            model_template=kwargs['model_template'],
            weights=kwargs['weights'][i]
        )

    end_time = time.time()
    elapsed = (end_time - start_time)
    s = 'TIME elapsed time = {}\n'.format(elapsed)
    record_processing(kwargs['reportfile'], s)


def aggregate_scales(**kwargs):
    # Reassemble the processed images from multiple scales
    pass



def main(**kwargs):
    # Take in a huge list of arguments
    # Pass each sub routine it's own little set of arguments

    repstr = 'Working on slide {}\n'.format(kwargs['filename'])
    record_processing(kwargs['reportfile'], repstr)

    exp_home = init_file_system(
        filename=kwargs['filename'],
        writeto=kwargs['writeto'],
        outputs=kwargs['outputs'],
        scales=kwargs['scales'],
        reportfile=kwargs['reportfile']
    )

    process_map = preprocessing(
        filename=kwargs['filename'],
        sub_dirs=kwargs['sub_dirs'],
        reportfile=kwargs['reportfile'],
        exp_home=exp_home
    )

    process_multiscale(
        filename=kwargs['filename'],
        sub_dirs=kwargs['sub_dirs'],
        scales=kwargs['scales'],
        weights=kwargs['weights'],
        reportfile=kwargs['reportfile'],
        process_map=process_map,
        exp_home=exp_home,
        weights=kwargs['weights']
    )



    aggregate_scales(args)
    pass



if __name__ == '__main__':
    # Take in or set args
    scales = [2100, 3500]
    weights = ''
    model_proto = ''
    outputs = [0,1,2,3,4]
    filename = ''
    main()
