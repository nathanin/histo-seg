#!/usr/bin/python
'''

Reassemble from tiles

'''
import gc
import numpy as np
import os
import sys
import cv2
from openslide import OpenSlide
histoseg_code = '/Users/nathaning/_projects/histo-seg/code'
sys.path.insert(0, histoseg_code)
import data
import generate_color

import time

# Set defaults
result_types = ['prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4']


def parse_dirs(scales, result_types=result_types):
    '''
    Returns: dict : {'result_types[0]':dirname, etc.}
    '''
    #dirlist = [os.path.join(imageroot, d) for d in os.listdir(imageroot)]
    dirlist = [d for d in os.listdir('.') if os.path.isdir(d)]
    retdict = {}
    for res in result_types:
        retdict[res] = [d for d in dirlist if res in d]

    # print retdict
    return retdict


def tile_wrt_top(svs, scale):
    leveldims = svs.level_dimensions
    downsample = svs.level_downsamples
    app_mag = svs.properties['aperio.AppMag']
    if app_mag == u'20':
        lvl20 = 0
    elif app_mag == u'40':
        lvl20 = 1

    ds_20 = downsample[lvl20]
    dims_20 = leveldims[lvl20]

    return int(
        np.round(leveldims[0][0] * scale / dims_20[0] /
                 np.sqrt(int(ds_20))))  # (EQ 1)


def get_ideal_pad(svs, m, level, tile_top):
    factor = int(
        svs.level_downsamples[level])  # This conversion isn't important
    lvl_dims = svs.level_dimensions[level]
    lvl_dims = lvl_dims[::-1]
    lvl_dims = (lvl_dims[0] / 2, lvl_dims[1] / 2)
    factor *= 2

    # Tilesize w.r.t. the target size (how big to make things)
    # This one can be fractional.
    # When it is fractional, and [r, c] ~ tilesize, there are problems
    tilesize = tile_top / factor
    tilesize_float = tile_top / float(factor)

    tilesR, tilesC = m.shape

    # X_c, and X_r is the projected size of the rebuilt image
    # We need them to know how much to resize and pad
    # Ignoring the fractional component of tilesize
    real_c = int(tilesC * tilesize)
    real_r = int(tilesR * tilesize)

    # Now get the padding w/r/t the ideal tile size
    ideal_c = int(tilesC * tilesize_float)
    ideal_r = int(tilesR * tilesize_float)

    #padc = lvl_dims[0] - ideal_c
    #padr = lvl_dims[1] - ideal_r

    dx = ideal_c / float(real_c)
    dy = ideal_r / float(real_r)

    # The question is ... can dx and dy be different?
    return tilesize, (dx, dy), (ideal_r, ideal_c)


def get_settings(svs, scales, svs_level, overlap=64):
    '''
    Returns a list the length of scales
    For each scale we need the expected reassembly size,
    the lowest level dimensions from the svs file

    # Use a constant level overlap of 64
    '''

    # Check number of svs levels against svs_level:
    #while svs_level >= svs.level_count: svs_level -= 1
    svs_low = svs.level_dimensions[svs_level]
    svs_low = svs_low[::-1]  # Flip b/c PIL is dumb
    svs_low = (svs_low[0] / 2, svs_low[1] / 2)

    # Loop over scales
    # Populate a list like:
    # settings = [[(scale1 size), (scale1 pad), (scale1 ts), (scale1 resize)],
    #             [(scale2 size), (scale2 pad), (scale2 ts), (scale2 resize)],
    #              etc...]
    settings = {}
    for scale in scales:
        # Initialize the output dict
        settings[scale] = []
        m = np.load('data_tilemap_{}.npy'.format(scale))  # TODO fix hardcode

        # Get all the elements for settings
        # Shape
        mshape = m.shape
        tile_top = tile_wrt_top(svs, scale)
        #overlap_top = tile_wrt_top(svs, overlap)

        # padding needed
        tilesize, dxdy, targetsize = get_ideal_pad(svs, m, svs_level, tile_top)

        # downsampled overlap size
        # downsampled overlap w.r.t. the fixed 256 tilesize on disk
        #ts = scale + 2 * overlap  # the actual tile size used
        #factor = ts / float(256)
        factor = (scale + 2 * overlap) / float(256)

        # Don't override `overlap` ; we're inside a loop
        ds_overlap = int(overlap / factor)

        # Build the output dict
        settings[scale].append(m)  # 0
        settings[scale].append(tilesize)  # 1
        settings[scale].append(dxdy)  # 2
        settings[scale].append(targetsize)  # 3
        settings[scale].append(svs_low)  # 4
        settings[scale].append(ds_overlap)  # 5

    return settings


def place_padding(img, target, value=0):
    # We have to implement this because it's desired behavior to add
    # padding on the ends only, not as a border
    # and i couldn't find something to do that within
    # base np or cv2
    imgr = img.shape[0]
    imgc = img.shape[1]
    targetr = target[0]
    targetc = target[1]
    r = targetr - imgr
    c = targetc - imgc

    padr = np.zeros(shape=(r, targetc, 3), dtype=img.dtype)
    padc = np.zeros(shape=(imgr, c, 3), dtype=img.dtype)

    # place in column pad first, then rows:
    img = np.hstack((img, padc))
    img = np.vstack((img, padr))

    return img


def rebuild(settings, dir_set, r):

    start_time = time.time()
    # unpack settings
    dirs = dir_set[r]

    # Track the rebuild images using a list
    scaleimgs = []

    # call the working function
    #def build_region(region,
    #                 m,
    #                 source_dir,
    #                 place_size,
    #                 overlap,
    #                 overlay_dir,
    #                 max_w=10000,
    #                 exactly=None,
    #                 pad=(0, 0)):

    # Unroll the loop for readability
    for s in settings.iterkeys():
        m, tilesize, dxdy, target_dim, svs_low, overlap = settings[s][:]
        region = [0, 0, m.shape[1], m.shape[0]]  # Flipped in data.*()
        source_dir = '{}_{}'.format(r, s)
        writename = '{}_{}.jpg'.format(r, s)
        print 'Working on {}'.format(writename)
        print 'Source_dir: {}'.format(source_dir)
        region = data.build_region(
            region=region,
            m=m,
            source_dir=source_dir,
            place_size=tilesize,
            overlap=overlap,
            overlay_dir='',
            max_w=None,
            exactly=target_dim)

        region = place_padding(region, svs_low)

        cv2.imwrite(filename=writename, img=region)

        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            #region = region[:,:,0]
        scaleimgs.append(region)

    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME reassemble.rebuild time: {}'.format(elapsed)

    return scaleimgs



def aggregate_scales(imgs, kernel=None, weights=None):
    # This is the function that averages across scale outputs
    # It gives back a single intensity image
     
    start_time = time.time()
    #if kernel is None:
    combo = [img for img in imgs]

    combo = np.dstack(combo)  # canon

    # Attempt to manage the memory a little
    del imgs
    del img
    gc.collect()

    if weights and len(weights) != combo.shape[2]:
        print 'WARNING Weights mismatching with image shape'
        while len(weights) < combo.shape[2]: weights.append(1.0)

    # Weights here.
    combo = np.average(combo, axis=2, weights=weights)

    ## Smooth again?

    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME reassemble.aggregate_scales time: {}'.format(elapsed)

    if kernel is None:
        combo = cv2.GaussianBlur(combo, (7,7), 0)
        return combo
    else:
        combo = cv2.GaussianBlur(combo, (7,7), 0)
        combo = cv2.morphologyEx(combo, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(combo, cv2.MORPH_OPEN, kernel)


def get_background(settings, rgb):
    # Define background
    # s0 = settings.keys()[0]
    # tilemap = settings[s0][0]
    # # tilemap = np.swapaxes(tilemap, 0,1)
    # background = tilemap == 0
    # background.dtype = np.uint8

    # # This is one of the most confounding quirks ever:
    # # For some reason, switch the 1st and 2nd elements of shape
    # background = cv2.resize(background, dsize=(rgb.shape[1], rgb.shape[0]),
    #     interpolation=cv2.INTER_NEAREST)


    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    background = cv2.GaussianBlur(rgb, (7,7), 0)
    bcg_level, background = cv2.threshold(background, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    background = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel)

    background.dtype = np.bool
    return background
    #background = np.swapaxes(background, 0,1)  # ????



def class_probs2labels(classimg, background=None, mode='argmax'):
    if mode == 'argmax':
        labelmask = np.argmax(classimg, axis=2) + 1

    elif mode == 'overlay':
        # Make a positive mask for each layer, decide:
        # G3 --> High Grade --> Benign --> Stroma
        # label_list = [cv2.cvtColor(classimg[:,:,x], cv2.COLOR_RGB2GRAY)
        #               for x in range(classimg.shape[2])]
        # label_list = [cv2.threshold(L, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #               for L in label_list]
        # label_list = [cv2.threshold(classimg[:,:,L], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #               for L in range(classimg.shape[2])] 
        # label_list = [cv2.threshold(classimg[:,:,x], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #               for x in range(classimg.shape[2])]

        labelmask = np.zeros(shape=label_list[0].shape, dtype=np.uint8)
        for k, L in enumerate(label_list):
            labelmask[L] = k+1

    if background is not None:
        if background.shape[:2] != labelmask.shape[:2]:
            x, y = labelmask.shape[:2]
            background.dtype = np.uint8
            background = cv2.resize(background, shape=(y, x),
                interpolation=cv2.INTER_NEAREST)
            background.dtype = np.bool

        labelmask[background] = 0

    return labelmask



def get_decision(classimg, svs, svs_level, colors, settings):
    # The main thing this function does now is to load the rgb
    # Load up RGB image
    rgb = svs.read_region(
        (0, 0), level=svs_level, size=svs.level_dimensions[svs_level])
    rgb = np.array(rgb)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
    #rgb = np.array(rgb)[:, :, :3]
    rgb = rgb[:, :, (2, 1, 0)]  # ???

    #TODO (nathan) only if the size is larger than some constant
    # -- Also have to change it somewhere else, idk where
    rgb = cv2.resize(rgb, dsize=(0,0), fx=0.5, fy=0.5)

    background = get_background(settings, rgb)

    labelmask = class_probs2labels(classimg, background=background, mode='argmax')
    # labelmask = np.argmax(classimg, axis=2) + 1
    # labelmask[background] = 0

    # Reassign Grade 5 to Grade 4. Re-label "high grade"
    # TODO (nathan) fix magic numbers !!!
    labelmask[labelmask == 5] = 2

    # Convert lable matrix to solid RGB colors
    #labelmask = impose_colors(labelmask, colors)

    # Check assertion that the images be the same shape.
    # at most, off by 1 px rounding error:
    if rgb.shape != classimg.shape:
        tt = classimg.shape[:2]  # Why is it flipped
        rgb = cv2.resize(rgb, dsize=(tt[1], tt[0]))

    #colorimg = data.overlay_colors(rgb, labelmask)
    colored_label = impose_colors(labelmask, colors)
    colorimg = data.overlay_colors(rgb, colored_label)

    return labelmask, colorimg


def impose_colors(label, colors):
    r = label.copy()
    g = label.copy()
    b = label.copy()

    u_labels = np.unique(label)
    for i, l in enumerate(u_labels):
        bin_l = label == l
        r[bin_l] = colors[l, 0]
        g[bin_l] = colors[l, 1]
        b[bin_l] = colors[l, 2]

    rgb = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 2] = r
    rgb[:, :, 1] = g
    rgb[:, :, 0] = b
    return rgb


def write_class_stats(labelimage, reportfile):
    repf = open(reportfile, 'a')

    factor_4x = 2.5

    # Tissue area, Epithelium area, High grade area, Low grade area
    high_grade = 1
    low_grade = 0
    #epith = [0,1,2]
    #tissue = [0,1,2,3]

    hg = (labelimage == high_grade).sum()
    lg = (labelimage == low_grade).sum()

    repf.write('REASSEMBLY AREA High Grade: {} px\n'.format(hg))
    repf.write('REASSEMBLY AREA Low Grade: {} px\n'.format(lg))

    repf.close()


def main(proj, svs, scales, scale_weights=None, ignorelabel = [0,4], reportfile = 'report.txt'):
    start_time = time.time()
    pwd = os.getcwd()
    workingdir = os.path.basename(svs)
    workingdir, _ = os.path.splitext(workingdir)
    svs = OpenSlide(svs)

    # now all the paths are going to be relative
    workingdir = os.path.join(proj, workingdir)
    os.chdir(workingdir)

    print 'Using reportfile {}'.format(reportfile)
    repf = open(reportfile, 'a')

    # Populate a list of dirs that contain things we want
    '''
    # dir_set[key] = [key_scale1, key_scale2, etc.]
    '''
    dir_set = parse_dirs(scales)

    svs_level = 4
    while svs_level >= svs.level_count:
        svs_level -= 1
    svs_level -= 1  # Use one less than the lowest level for better res.
    settings = get_settings(svs, scales, svs_level)

    # scaleimgs is a list of lists:
    # e.g. scaleimgs[0] is images of class 0 from all scales
    scaleimgs = [rebuild(settings, dir_set, r) for r in result_types]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    colors = generate_color.generate(
        n=len(result_types), whiteidx=ignorelabel, cmap='brg')

    repstr = 'COLORS:'
    for k in range(colors.shape[0]):
        repstr = '{}\n{:3d} | {}'.format(repstr, k, colors[k, :])

    repstr = '{}\n'.format(repstr)
    repf.write(repstr)

    # Aggregate from the assembled files
    classimg = []
    for c, r in zip(scaleimgs, result_types):
        filename = 'combo_{}.jpg'.format(r)

        # aggregate_scales() essentially averages the elements of c
        combo = aggregate_scales(c, kernel=kernel, weights=scale_weights)

        classimg.append(combo)  # Important to go in order
        cv2.imwrite(filename=filename, img=combo)

    # Get a background image from tilemaps
    # Do the final classification
    classimg = np.dstack(classimg)

    # TODO (nathan) put a colormap here to handle more than 4 channels
    cv2.imwrite(filename='dev_rgba.png', img=classimg[:,:,:4])

    # Make decisions and overlay discrete classes
    labelimage, colorimg = get_decision(classimg, svs, svs_level, colors, settings)

    repf.close()
    write_class_stats(labelimage, reportfile)
    repf = open(reportfile, 'a')

    labelimage_name = 'label.png'
    cv2.imwrite(filename=labelimage_name, img=labelimage)

    color_filename = 'color.jpg'
    cv2.imwrite(filename=color_filename, img=colorimg)

    os.chdir(pwd)  # Change back
    # TODO (nathan) implement cleanup

    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME reassemble.main file: {} time: {}'.format(
        workingdir, elapsed)

    repf.write('TIME REASSEMBLY {}\n'.format(elapsed))
    repf.close()

    return labelimage, cv2.cvtColor(colorimg, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    #start_time = time.time()
    proj = sys.argv[1]
    svs = sys.argv[2]

    #proj = '/home/nathan/histo-seg/pca/seg_0.8.1024_resume'
    #svs = sys.argv[1]
    #svs = '/home/nathan/data/pca_wsi/1305400.svs'
    print 'Working on image: {}'.format(svs)
    print 'Reading and writing to {}'.format(proj)

    # The strategy for weighting is to have scales not explicitly
    # included in the training weighed less
    scales = [2100, 3000]
    #scales = [364, 384]
    scale_weights = [0.5, 1]  # TODO (nathan)

    main(proj, svs, scales, scale_weights)

    #end_time = time.time()
    #elapsed = (end_time - start_time)
    #repf = open(reportfile, 'a')
    #repf.write('TIME ASSEMBLY TOTAL {}\n'.format(elapsed))
    #repf.close()

