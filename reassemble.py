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

# Set defaults
result_types = ['prob0', 'prob1', 'prob2', 'prob3', 'prob4']


def parse_dirs(scales, result_types=result_types):
    '''
    Returns: dict : {'result_types[0]':dirname, etc.}
    '''
    #dirlist = [os.path.join(imageroot, d) for d in os.listdir(imageroot)]
    dirlist = [d for d in os.listdir('.') if os.path.isdir(d)]
    retdict = {}
    for res in result_types:
        retdict[res] = [d for d in dirlist if res in d]

    print retdict
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
            region = region[:,:,0]
        scaleimgs.append(region)

    return scaleimgs


# TODO (nathan) weighting
def aggregate_scales(imgs, kernel=None, weights=None):
    #if kernel is None:
    combo = [img for img in imgs]
    #else:
        # cv2 insists on returning 3-channel images
        #combo = [cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)[:,:,0] for img in imgs]

    # Weights here.
    combo = np.dstack(combo)  # canon
    #combo = np.argmax(combo,
    del imgs
    del img
    gc.collect()

    if weights and len(weights) != combo.shape[2]:
        print 'WARNING Weights mismatching with image shape'
        while len(weights) < combo.shape[2]: weights.append(1.0)

    combo = np.average(combo, axis=2, weights=weights)

    ## Smooth again?
    if kernel is None:
        return combo
    else:
        return cv2.morphologyEx(combo, cv2.MORPH_OPEN, kernel)


def decision(classimg, svs, svs_level, colors):
    rgb = svs.read_region(
        (0, 0), level=svs_level, size=svs.level_dimensions[svs_level])
    rgb = np.array(rgb)[:, :, :3]
    rgb = rgb[:, :, (2, 1, 0)]
    rgb = cv2.resize(rgb, dsize=(0,0), fx=0.5, fy=0.5)

    classimg = np.argmax(classimg, axis=2)
    classimg = impose_colors(classimg, colors)


    # Final assertion that the images be the same shape.
    # at most, off by 1 px rounding error:
    if rgb.shape != classimg.shape:
        tt = classimg.shape[:2]  # Why is it flipped
        rgb = cv2.resize(rgb, dsize=(tt[1], tt[0]))

    colorimg = data.overlay_colors(rgb, classimg)

    return classimg, colorimg


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


def main(proj, svs, scales, scale_weights=None):
    # Set some constant
    pwd = os.getcwd()
    #svs = '{}.svs'.format(imageroot)
    workingdir = os.path.basename(svs)
    workingdir, _ = os.path.splitext(workingdir)
    svs = OpenSlide(svs)

    # Do the thing I used to do in Matlab:
    # now all the paths are going to be relative
    workingdir = os.path.join(proj, workingdir)
    os.chdir(workingdir)

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
    '''
    # Now ready to do reassembly; re-use ~/histo-seg/code/data.py
    # Pass in proper settings to assemble_region:
    # Do it like this:
    #   build output 1, all scales
    #   build output 2, all scales,
    #   etc.
    # :: loop over result_types
    '''
    scaleimgs = [rebuild(settings, dir_set, r) for r in result_types]
    '''
    # scaleimgs is like:
    #   [[c0_s1, c0_s2, c0_s3],
    #    [c1_s1, c1_s2, s1_s3],
    #    etc. ]
    # Now combine each class
    # With a little smoothing and weighting
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    colors = generate_color.generate(
        n=len(result_types), whiteidx=3, cmap='brg')

    # Aggregate from the assembled files
    # Fix scale_weights
    #if scale_weights is None:
    #    scale_weights = [1] * len(scales)
    classimg = []
    for c, r in zip(scaleimgs, result_types):
        filename = 'combo_{}.jpg'.format(r)
        combo = aggregate_scales(c, kernel=kernel, weights=scale_weights)
        classimg.append(combo)  # Important to go in order
        cv2.imwrite(filename=filename, img=combo)

    # Do the final classification
    classimg = np.dstack(classimg)

    # Write out something pretty
    cv2.imwrite(filename='dev_rgba.jpg', img=classimg[:,:,:4])

    # Make decisions and overlay discrete classes
    classimg, colorimg = decision(classimg, svs, svs_level, colors)
    classfilename = 'class.png'
    colorfilename = 'color.jpg'
    cv2.imwrite(filename=classfilename, img=classimg)
    cv2.imwrite(filename=colorfilename, img=colorimg)

    os.chdir(pwd)  # Change back
    # TODO (nathan) implement cleanup


if __name__ == '__main__':
    proj = '/home/nathan/histo-seg/pca/seg_0.8'
    svs = sys.argv[1]
    #svs = '/home/nathan/data/pca_wsi/1305471.svs'
    print 'Working on image: {}'.format(svs)
    print 'Reading and writing to {}'.format(proj)

    # The strategy for weighting is to have scales not explicitly
    # included in the training weighed less
    scales = [384, 512, 600, 656]
    scale_weights = [2, 0.75, 1, 0.5]  # TODO (nathan)

    main(proj, svs, scales, scale_weights)
