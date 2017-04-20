import data
import histoseg
import reassemble

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

    # Sub-dirs hold tiles and the results

    if os.path.exists(exp_home):
        shutil.rmtree(exp_home)

    try:
        os.makedirs(exp_home)
        reportfile = os.path.join(exp_home, 'report.txt')
        s = 'Creating file system at : {}\n'.format(exp_home)
        s = '{}Recording output to {}\n'.format(s, reportfile)
        record_processing(reportfile, s)

        for scale in kwargs['scales']:
            d = os.path.join(exp_home, 'tiles_{}'.format(scale))
            os.makedirs(d)
            for d in kwargs['outputs']:
                d = os.path.join(exp_home, 'prob_{}_{}'.format(d,scale))
                os.makedirs(d)
                repstr = 'Created {}\n'.format(d)
                record_processing(reportfile, repstr)
    except:
        print 'Error initializing filesystem'
        print 'Attempting to create {}'.format(exp_home)
        print 'Attempting to create {}'.format(d)
        s = 'Failure intializing filesystem\n'.format(exp_home)
        record_processing(reportfile, s)
        return 0

    return exp_home, reportfile


def whitespace(img, reportfile, white_pt=190):
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    if n_masks == 1:
        mask = masks[0]
    elif n_masks > 1:
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

    lo_res = 128
    masks = [whitemap]
    process_map = get_process_map(img, lo_res, masks, kwargs['reportfile'])

    return process_map


def clean_intermediates(dirlist):
    # Bludgeon away the details
    pass


def tile_scale(**kwargs):
    # Perform tiling for one scale, according to the usual method.
    # Call data.tile_wsi
    writeto = 'tiles_{}'.format(kwargs['tilesize'])

    # Work with kwargs['process_map'] to get tilemap:
    wsi = kwargs['wsi']
    tilesize = kwargs['tilesize']
    if wsi.properties['aperio.AppMag'] == '20':
        level_20 = 0
    elif wsi.properties['aperio.AppMag'] == '40':
        level_20 = 1
    nrow, ncol = [wsi.level_dimensions[level_20][0] / tilesize,
                  wsi.level_dimensions[level_20][1] / tilesize]
    tilemap = kwargs['process_map']
    tilemap.dtype = np.uint8
    tilemap = cv2.resize(tilemap,
                         dsize=(nrow,ncol),
                         interpolation=cv2.INTER_NEAREST)
    s = 'Resized process_map from {} to {}\n'.format(
        kwargs['process_map'].shape, tilemap.shape
    )
    record_processing(kwargs['reportfile'], s)

    tilemap = data.tile_wsi(
        wsi=kwargs['wsi'],
        tilesize=kwargs['tilesize'],
        writesize=kwargs['writesize'],
        writeto=os.path.join(kwargs['exp_home'], writeto),
        overlap=kwargs['overlap'],
        prefix=kwargs['prefix'],
        tilemap=tilemap
    )

    tilemap_name = os.path.join(kwargs['exp_home'],
                                'data_tilemap_{}.npy'.format(kwargs['tilesize']))
    s = 'Saving tilemap to {}\n'.format(tilemap_name)
    record_processing(kwargs['reportfile'], s)
    np.save(file=tilemap_name, arr=tilemap)


def process_scale(**kwargs):
    # This function is basically the main one
    s = 'Processing scale {}\n'.format(kwargs['scale'])
    s = '{}Beginning to tile... \n'.format(s)
    record_processing(kwargs['reportfile'],s)

    # Some hard coded constants in there.. that's OK for now
    tile_start = time.time()
    tile_scale(wsi=kwargs['wsi'], tilesize=kwargs['scale'], writesize=256,
               overlap=64, prefix='tile', process_map=kwargs['process_map'],
               reportfile=kwargs['reportfile'], exp_home=kwargs['exp_home'])

    # Record the timing info
    tile_end = time.time()
    tile_elapsed = (tile_end - tile_start)
    s = 'TIME Tile scale {} elapsed = {}\n'.format(
        kwargs['scale'], tile_elapsed)
    record_processing(kwargs['reportfile'], s)

    # Re-define the appropriate output dirs:
    expdirs = [os.path.join(kwargs['exp_home'],
                            'tiles_{}'.format(kwargs['scale']))]
    for output in kwargs['outputs']:
        d = os.path.join(kwargs['exp_home'],
                         'prob_{}_{}'.format(output, kwargs['scale']))
        if os.path.exists(d):
            expdirs.append(d)
        else:
            raise Exception('Path exception: {} does not exist'.format(d))

    # Call histoseg.process()
    s = 'Passing control to histoseg.process\n'
    record_processing(kwargs['reportfile'], s)

    process_start = time.time()
    histoseg.process(
        exphome=kwargs['exp_home'],
        expdirs=expdirs,
        model_template=kwargs['model_template'],
        weights=kwargs['weights'],
        mode=0,
        GPU_ID=0,
        reportfile=kwargs['reportfile']
    )
    process_end = time.time()
    process_elapsed = (process_end - process_start)
    s = 'TIME Processing scale {} elapsed = {}\n'.format(
        kwargs['scale'], process_elapsed)
    record_processing(kwargs['reportfile'], s)


def process_multiscale(**kwargs):
    s = 'Working on multiscale processing\n'
    record_processing(kwargs['reportfile'], s)

    wsi = OpenSlide(kwargs['filename'])

    start_time = time.time()
    for i,scale in enumerate(kwargs['scales']):
        process_scale(
            scale=scale,
            wsi=wsi,
            process_map=kwargs['process_map'],
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

    assembly_start = time.time()
    # Call reassemble.main()
    s = 'Passing control to reassemble.main()\n'
    record_processing(kwargs['reportfile'], s)
    labels, colorized = reassemble.main(
        proj=kwargs['project'],
        svs=kwargs['filename'],
        scales=kwargs['scales'],
        scale_weights=kwargs['scale_weights'])

    assembly_end = time.time()
    assembly_elapsed = (assembly_end - assembly_start)
    s = 'TIME Assembly elapsed = {}\n'.format(assembly_elapsed)
    record_processing(kwargs['reportfile'], s)

    return labels, colorized


def convert_px2micron(px, conversion=5.0):
    micron_sq = px / conversion
    return np.sqrt(micron_sq)


def analyze_result(**kwargs):
    # Low grade index = 1
    # High grade index = 2
    # Benign index = 3
    # Stroma index = 4
    stats = {'Cancer_area':0,
             'Low_grade_area':0,
             'High_grade_area':0,
             'Low_grade_percent':0,
             'High_grade_percent':0,
             'Tissue_area':0}
    labels = kwargs['labels']

    # Total cancer area
    canc_area = np.add([(labels == 1).sum(),
                        (labels == 2).sum()])
    stats['Cancer_area'] = convert_px2micron(canc_area)
    s = 'Analysis: Cancer Area = {}\n'.format(canc_area)
    record_processing(kwargs['reportfile'], s)

    # Grade areas
    low_grade_area = (labels == 1).sum()
    high_grade_area = (labels == 2).sum()
    stats['Low_grade_area'] = convert_px2micron(low_grade_area)
    stats['High_grade_area'] = convert_px2micron(high_grade_area)
    s = 'Analysis: Low Grade Area = {}\n'.format(low_grade_area)
    s = '{}Analysis: High Grade Area = {}\n'.format(s, high_grade_area)
    record_processing(kwargs['reportfile'], s)

    # Grade percentages
    low_grade_percent = low_grade_area / float(canc_area)
    high_grade_percent = high_grade_area / float(canc_area)
    stats['Low_grade_percent'] = low_grade_percent
    stats['High_grade_percent'] = high_grade_percent
    s = 'Analysis: Low Grade Percent = {}\n'.format(low_grade_percent)
    s = '{}Analysis: High Grade Percent = {}\n'.format(s, high_grade_percent)
    record_processing(kwargs['reportfile'], s)

    # Tissue area
    tiss_area = np.add([(labels == 1).sum(),
                        (labels == 2).sum(),
                        (labels == 3).sum(),
                        (labels == 4).sum()])
    stats['Tissue_area'] = convert_px2micron(tiss_area)
    s = 'Analysis: Tissue Area = {}\n'.format(tiss_area)
    record_processing(kwargs['reportfile'], s)
    return stats


def create_report(**kwargs):
    time_elapsed = kwargs['time_elapsed']
    pass


def main(**kwargs):
    # Take in a huge list of arguments
    # Pass each sub routine it's own little set of arguments

    time_all_start = time.time()
    exp_home, reportfile = init_file_system(
        filename=kwargs['filename'],
        writeto=kwargs['writeto'],
        outputs=kwargs['outputs'],
        scales=kwargs['scales']
    )
    #reportfile = os.path.join(exp_home, 'report.txt')
    print 'Recording run info to {}'.format(reportfile)
    repstr = 'Working on slide {}\n'.format(kwargs['filename'])
    record_processing(reportfile, repstr)

    process_map = preprocessing(
        filename=kwargs['filename'],
        reportfile=reportfile,
    )

    process_multiscale(
        filename=kwargs['filename'],
        scales=kwargs['scales'],
        weights=kwargs['weights'],
        outputs=kwargs['outputs'],
        model_template=kwargs['model_template'],
        reportfile=reportfile,
        process_map=process_map,
        exp_home=exp_home
    )

    labels, colorized = aggregate_scales(
        project=kwargs['writeto'],
        filename=kwargs['filename'],
        scales=kwargs['scales'],
        scale_weights=kwargs['scale_weights'],
        reportfile=reportfile
    )

    time_all_end = time.time()
    time_total_elapsed = (time_all_end - time_all_start)
    s = 'TIME total elapsed = {}\n'.format(time_total_elapsed)
    record_processing(reportfile, s)

    # TODO (nathan) create some slide-level output
    analyze_result(
        filename=kwargs['filename'],
        exp_home=exp_home,
        labels=labels,
        colorized=colorized,
        reportfile=reportfile
    )


if __name__ == '__main__':
    # Take in or set args
    # These stay the same
    scales = [512, 1024]
    scale_weights = [1, 0.5]
    weights = ['/home/nathan/semantic-pca/weights/seg_0.8.1/norm_resumed_iter_32933.caffemodel',
               '/home/nathan/semantic-pca/weights/seg_0.8.1024/norm_iter_125000.caffemodel']
    model_template = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'
    writeto = '/home/nathan/histo-seg/pca/dev'
    outputs = [0,1,2,3,4]

    # Filename changes
    #filename = sys.argv[1]
    filename = '/home/nathan/data/pca_wsi/1305400.svs'
    main(filename=filename,
         scales=scales,
         scale_weights=scale_weights,
         weights=weights,
         model_template=model_template,
         outputs=outputs,
         writeto=writeto)
