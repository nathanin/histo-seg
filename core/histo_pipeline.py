#!/usr/local/bin/python

import data
import histoseg
import reassemble

import os
import glob
import time
import datetime
import shutil
import sys

import cv2
import numpy as np
from openslide import OpenSlide


# For making reports
from matplotlib import pyplot as plt
from matplotlib.table import Table
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

rcParams.update({'figure.autolayout': True})

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


# Two helper functions that let me skip a lot of effort
def get_exp_home(writeto, filename):
    tail = os.path.basename(filename)
    slide_name, ex = os.path.splitext(tail)
    exp_home = os.path.join(writeto, slide_name)

    return exp_home


def get_reportfile(exp_home, reportfile='report.txt'):
    return os.path.join(exp_home, reportfile)


def init_file_system(**kwargs):
    # Create the file system
    exp_home = get_exp_home(kwargs['writeto'], kwargs['filename'])
    # filename = kwargs['filename']
    # tail = os.path.basename(filename)
    # slide_name, ex = os.path.splitext(tail)
    # exp_home = os.path.join(kwargs['writeto'], slide_name)

    # Sub-dirs hold tiles and the results

    if os.path.exists(exp_home):
        shutil.rmtree(exp_home)

    try:
        os.makedirs(exp_home)
        reportfile = get_reportfile(exp_home)
        # reportfile = os.path.join(exp_home, 'report.txt')
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


def whitespace(img, reportfile, white_pt=210):
    # Simple. Could be more sophisticated
    background = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # background = cv2.GaussianBlur(img, (7,7), 0)
    bcg_level, background = cv2.threshold(background, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    background = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel)

    background.dtype = np.bool

    s = 'Found {} px above grayscale value {}\n'.format(
        background.sum(), bcg_level
    )
    record_processing(reportfile, s)

    return background


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

    return mask


def preprocessing(**kwargs):
    wsi = OpenSlide(kwargs['filename'])

    img = read_region(wsi,
                      (0,0),
                      wsi.level_count-1,
                      wsi.level_dimensions[-1]
                      )
    s = 'PREPROCESS Successfully read image from {}\n'.format(kwargs['filename'])
    record_processing(kwargs['reportfile'], s)

    # Boolean image of white areas
    whitemap = whitespace(img, kwargs['reportfile'])

    # TODO (nathan) add tumor location here
    #unprocessable = unprocessable_area(img)
    # I don't think an H& E will really do it. 
    # It'll take some mighty processing to get the job done
    # I mean like.. it's probabaly doable.
    # OK I'll try tomorrow.

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
    s = 'TILE Resized process_map from {} to {}\n'.format(
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
    s = 'TILE Saving tilemap to {}\n'.format(tilemap_name)
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
    s = '*********\tPassing control to histoseg.process\n'
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
    s = '*********\tPassing control to reassemble.main()\n'
    record_processing(kwargs['reportfile'], s)
    labels, colorized, classimg, colormap = reassemble.main(
        proj=kwargs['project'],
        svs=kwargs['filename'],
        scales=kwargs['scales'],
        scale_weights=kwargs['scale_weights']
    )

    assembly_end = time.time()
    assembly_elapsed = (assembly_end - assembly_start)
    s = 'TIME Assembly elapsed = {}\n'.format(assembly_elapsed)
    record_processing(kwargs['reportfile'], s)

    return labels, colorized, classimg, colormap


def convert_px2micron(px, conversion=4.0):
    micron_sq = px / conversion
    return np.sqrt(micron_sq)


def px2um_factor(wsi, levelX):
    if wsi.properties['aperio.AppMag'] == '40':
        lvl0_factor = 0.25 
    elif wsi.properties['aperio.AppMag'] == '20':
        lvl0_factor = 0.5
    else: 
        raise Exception('OpenSlide or SVS file error')

    lvl0_x = wsi.dimensions[0]
    ratio = lvl0_x / levelX  # should be some power of 2

    return lvl0_factor * ratio


def generate_stats(**kwargs):
    # Unprocessed = 0
    # Low grade index = 1
    # High grade index = 2
    # Benign index = 3
    # Stroma index = 4

    stats = {'Tissue_area':0,
             'Cancer_area':0,
             'Low_grade_area':0,
             'High_grade_area':0}
    labels = kwargs['labels']

    # Take care of me being stupid first:
    process_map = kwargs['process_map']
    process_map.dtype = np.uint8
    if process_map.shape[:2] != labels.shape[:2]:
        print 'ANALYSIS Resizing processmap to match labels'
        process_map = cv2.resize(process_map, dsize=labels.shape[:2],
            interpolation=cv2.INTER_NEAREST)

    wsi = OpenSlide(kwargs['filename'])
    conversion = px2um_factor(wsi, process_map.shape[0])

    s = 'ANALYSIS: Factor for pixel to micron conversion: {} micron/pixel\n'.format(conversion)
    record_processing(kwargs['reportfile'], s)

    # Estimated total tissue:
    tissue_area = process_map.sum()
    stats['Tissue_area'] = tissue_area
    s = 'ANALYSIS: Analyzed Area = {}\n'.format(tissue_area)
    record_processing(kwargs['reportfile'], s)

    # Total cancer area
    cancer_area = np.add((labels == 1).sum(),
                       (labels == 2).sum())
    stats['Cancer_area'] = cancer_area
    s = 'ANALYSIS: Cancer Area = {}\n'.format(cancer_area)
    record_processing(kwargs['reportfile'], s)

    # Grade areas
    low_grade_area = (labels == 1).sum()
    high_grade_area = (labels == 2).sum()
    stats['Low_grade_area'] = low_grade_area
    stats['High_grade_area'] = high_grade_area
    s = 'ANALYSIS: Low Grade Area = {}\n'.format(low_grade_area)
    s = '{}ANALYSIS: High Grade Area = {}\n'.format(s, high_grade_area)
    record_processing(kwargs['reportfile'], s)

    return stats



def draw_class_images(classimg, exp_home):
    ax = plt.figure(figsize=(5,9))
    c_dict = {0: 'G3', 1: 'G4', 2: 'BN', 3: 'ST', 4: 'G5'}
    nclass = classimg.shape[2]
    for k in range(nclass):
        c = classimg[:,:,k]
        c = cv2.resize(c, dsize=(0,0), fx=0.5, fy=0.5,
            interpolation=cv2.INTER_LINEAR)

        plt.subplot(nclass, 2, k*2 + 1)
        _ = plt.hist(c.ravel(), normed=1, bins=40)
        plt.ylabel('frequency')
        plt.xlabel('probability')

        plt.subplot(nclass, 2, k*2 + 2)
        plt.imshow(c, cmap='Greys')
        plt.xticks([]), plt.yticks([])
        plt.title('Class {}'.format(c_dict[k]))
        plt.colorbar()

    plt.savefig(os.path.join(exp_home, 'class_images.pdf'))
    plt.close()


def imgs_to_plot(img):
    # make background 0 where there is __no tissue__
    background = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # background = cv2.GaussianBlur(background, (7,7), 0)
    bcg_level, background = cv2.threshold(background, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel)
    background = 255- background

    # Reimplement data.get_all_regions, except do it better
    contours = []
    cntrs,x = cv2.findContours(image=background, mode=cv2.RETR_EXTERNAL,
                           method=cv2.CHAIN_APPROX_SIMPLE)

    # Make sure there's at least one:
    n = 200
    while len(contours) == 0:
        thrsh = n ** 2
        for cc in cntrs:
            if cv2.contourArea(cc) > thrsh:
                contours.append(cc)

        n -= 50

    def cutout_region(cc):
        bbox = cv2.boundingRect(cc)
        area = cv2.contourArea(cc)
        x = bbox[0]
        y = bbox[1]
        dx = x+bbox[2]
        dy = y+bbox[3]
        simg = img[y:dy, x:dx, :]

        ### If rows > cols, flip it:
        rows, cols = simg.shape[:2]
        if rows > cols:
        #     # M = cv2.getRotationMatrix2D((rows/2, cols/2), 90, 1)
        #     # simg = cv2.warpAffine(simg, M, (cols, rows))

        #     # For 90deg use transpose
        #     # Can't use simg.T since it's 3-channel
            simg = np.swapaxes(simg,0,1)

        return simg, area

    regions, areas = zip(*(cutout_region(cc) for cc in contours))

    # Sort by cols
    def sort_regions(regions, areas):
        reg_cols = [r.shape[1] for r in regions]
        sort_indices = np.argsort(reg_cols)

        reg_sorted = [regions[k] for k in sort_indices]
        areas_sorted = [areas[k] for k in sort_indices]
        return reg_sorted, areas_sorted

    regions, areas = sort_regions(regions, areas)

    return regions, areas
    

def build_stat_string(**kwargs):
    pattern_dict = {0: 'G3', 1: 'G4+G5'}

    # Constant header
    header = [
        'Slide\nName', 
        'Processing\nTime (s)', 
        'Date', 
        'Tumor %', 
        'Major\nPatterns'
    ]

    # Lucky that data is just one row
    slide_name = os.path.basename(kwargs['filename'])

    # Processing time 
    time_min = np.floor(kwargs['time_elapsed'] / 60)
    time_sec = kwargs['time_elapsed'] % 60

    # Tumor Area
    stats = kwargs['stats']
    cancer_area = stats['Cancer_area']
    Tumor_pct = cancer_area / float(stats['Tissue_area'])

    # Major patterns: 
    pattern_threshold = cancer_area * 0.1  # 10% cancer area
    grade_areas = [stats['Low_grade_area'], stats['High_grade_area']]

    pattern_str = r''
    for k, ga in enumerate(grade_areas):
        if ga > pattern_threshold:
            pattern_str = r'{} {} $({:3.2f})$\%'.format(
                pattern_str,
                pattern_dict[k],
                ga / float(cancer_area) * 100)

    wsi = OpenSlide(kwargs['filename'])
    process_map = kwargs['process_map']

    data = [
        slide_name,
        '${}min$ ${:3.1f}s$'.format(int(time_min), time_sec),
        r'{}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        r'${:3.2f}$\%'.format(Tumor_pct * 100),
        pattern_str
    ]


    return header, data



def create_report(**kwargs):
    draw_class_images(kwargs['classimg'], kwargs['exp_home'])
    reportfile = os.path.join(kwargs['exp_home'], 'report.pdf')
    colors = kwargs['colors']

    regions, areas = imgs_to_plot(kwargs['colorized'])
    nreg = len(regions)

    with PdfPages(reportfile) as pdf:
        if nreg == 1:
            ax=plt.figure()
            plt.imshow(regions[0])
            # ax.set_xlabel([])
            # ax.set_ylabel([])
            plt.title('Region 1, Area = {}'.format(areas[0]))
            pdf.savefig(dpi=300)
            plt.close()
        else:
            for k, (reg, area) in enumerate(zip(regions, areas)):
                ax=plt.figure()
                plt.imshow(reg)
                # ax.set_xlabel([])
                # ax.set_ylabel([])
                plt.title('Region {}, Area = {}'.format(k, area))
                pdf.savefig(dpi=300)
                plt.close()

        plt.rc('text', usetex=True)
        fig, ax = plt.subplots(1,1)
        # ax.set_xtick([])
        # ax.set_ytick([])
        bbox_colors = [0.1, 0.1, 0.2, 0.8]
        tb = Table(ax, bbox=bbox_colors)
        # tb = Table(ax)
        tb.add_cell(1, 1, 0.1, 0.25, text='BG', loc='center', facecolor=colors[0,:]/255.0)
        tb.add_cell(2, 1, 0.1, 0.25, text='Low Grade',loc='center', facecolor=colors[1,:]/255.0)
        tb.add_cell(3, 1, 0.1, 0.25, text='High Grade',loc='center', facecolor=colors[2,:]/255.0)
        tb.add_cell(4, 1, 0.1, 0.25, text='BN',loc='center', facecolor=colors[3,:]/255.0)
        tb.add_cell(5, 1, 0.1, 0.25, text='ST',loc='center', facecolor=colors[4,:]/255.0)
     
        ax.add_table(tb)

        header, data = build_stat_string(
            filename=kwargs['filename'],
            time_elapsed=kwargs['time_elapsed'],
            stats=kwargs['stats'],
            process_map=kwargs['process_map']
        )

        bbox_info = [0.5, 0.1, 0.4, 0.8]
        tb = Table(ax, bbox=bbox_info)
        nrow = float(len(data))
        for k, (h, d) in enumerate(zip(header, data)):
            tb.add_cell(k, -1, 0.1, 1/nrow, text=h)
            tb.add_cell(k, 1, 0.3, 1/nrow, text=d)

        # tab = ax.table(cellText=data, rowLabels=header, cellLoc='center',
        #     bbox=bbox_info)
        # tab = ax.table(cellText=data, rowLabels=header, loc='center', cellLoc='center')

        ax.add_table(tb)
        pdf.savefig()
        plt.close()


def main(**kwargs):
    # Take in a huge list of arguments
    # Pass each sub routine it's own little set of arguments

    time_all_start = time.time()

    exp_home = get_exp_home(kwargs['writeto'], kwargs['filename'])
    reportfile = get_reportfile(exp_home)

    # exp_home, reportfile = init_file_system(
    #     filename=kwargs['filename'],
    #     writeto=kwargs['writeto'],
    #     outputs=kwargs['outputs'],
    #     scales=kwargs['scales']
    # )

    print 'Recording run info to {}'.format(reportfile)
    repstr = 'Working on slide {}\n'.format(kwargs['filename'])
    record_processing(reportfile, repstr)

    process_map = preprocessing(
        filename=kwargs['filename'],
        reportfile=reportfile
    )

    # process_multiscale(
    #     filename=kwargs['filename'],
    #     scales=kwargs['scales'],
    #     weights=kwargs['weights'],
    #     outputs=kwargs['outputs'],
    #     model_template=kwargs['model_template'],
    #     reportfile=reportfile,
    #     process_map=process_map,
    #     exp_home=exp_home
    # )

    # # # In dev mode it's ok to just do this; it's pretty quick
    labels, colorized, classimg, colormap = aggregate_scales(
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

    stats = generate_stats(
        filename=kwargs['filename'],
        exp_home=exp_home,
        labels=labels,
        process_map=process_map,
        reportfile=reportfile
    )

    create_report(
        filename=kwargs['filename'],
        exp_home=exp_home,
        colorized=colorized,
        time_elapsed=time_total_elapsed,
        process_map=process_map,
        stats=stats,
        classimg=classimg,
        colors=colormap
    )

    ## Some stdout feedback
    print '\nFinished processing {}'.format(kwargs['filename'])
    print 'Time elapsed: {:0.2f}s'.format(time_total_elapsed)


if __name__ == '__main__':
    # Take in or set args
    # These stay the same
    scales = [384, 512, 896]
    scale_weights = [3, 0.5, 0.25]
    # scales = [384, 896]
    # scale_weights = [3, 0.25]
    weights = ['/home/nathan/semantic-pca/weights/seg_0.8.1/norm_resumed_iter_32933.caffemodel',
               '/home/nathan/semantic-pca/weights/seg_0.5/norm_iter_125000.caffemodel',
               '/home/nathan/semantic-pca/weights/seg_0.8.1024/norm_iter_125000.caffemodel']
    model_template = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'
    #writeto = '/Users/nathaning/_projects/histo-seg/pca/dev'
    writeto = '/home/nathan/histo-seg/pca/dev'
    outputs = [0,1,2,3,4]

    
    filename = sys.argv[1]
    #filename = '/Users/nathaning/_projects/histo-seg/pca/dev/1305497.svs'
    # filename = '/home/nathan/data/pca_wsi/NEPC12E-HE.svs'
    main(filename=filename,
         scales=scales,
         scale_weights=scale_weights,
         weights=weights,
         model_template=model_template,
         outputs=outputs,
         writeto=writeto)
