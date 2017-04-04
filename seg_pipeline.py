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


# Define inspection code that spits out the line it's called from (as str)
def PrintFrame():
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    thisfile = info.filename
    thisfun = info.function
    thisline = info.lineno
    return '{} in {} (@ line {})'.format(thisfile, thisfun, thisline)


##################################################################
##################################################################
###
###       ~~~~~~~~ functions to do work ~~~~~~~~
###
##################################################################
##################################################################


def run_histoseg(exphome, expdirs, weights, model_template, mode, GPU_ID, dev):
    # Echo inputs
    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tRunning histoseg.process: '
    print '\tSource: {}'.format(expdirs[0])
    print '\tDestination: {}'.format(expdirs[1:])
    print '\tModel template: {}'.format(model_template)
    print '\tWeights: {}'.format(weights)
    print '\tMode: {}'.format(mode)
    print '\tRunning on GPU: {}'.format(GPU_ID)

    if dev:
        print '\nRUNNING DEVELOPMENT MODE'
        histoseg.process_dev(expdirs)
    else:
        histoseg.process(exphome, expdirs, model_template, weights, mode,
                         GPU_ID)


def make_data_inference(filename,
                        writeto,
                        create,
                        tilesize,
                        writesize,
                        overlap=0,
                        remove_first=False):
    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tRunning data creation for inference:'
    print '\tFile: {}'.format(filename)
    print '\tDestination: {}'.format(writeto)
    return data.make_inference(filename, writeto, create, tilesize, writesize,
                               overlap, remove_first)


def assemble_tiles(result_root, expdirs, writesize, overlap, overlay, filename,
                   tilesize):
    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tAssembling tiles:'
    print '\tSaving result to : {}'.format(result_root)
    area_cutoff = data.calc_tile_cutoff(filename, tilesize)
    data.assemble(result_root, expdirs, writesize, overlap, overlay,
                  area_cutoff, tilesize)


def cleanup(dirs):
    for d in dirs:
        if d != '/':
            print 'Cleaning {}'.format(d)
            shutil.rmtree(d)


def get_downsample_overlap(tilesize, writesize, overlap):
    ts = tilesize + 2 * overlap
    factor = ts / float(writesize)

    return int(overlap / factor)


def run_inference(do_clean=True, do_parsing=True, do_assembly=True, **kwargs):
    if do_parsing:
        args = parse_options(**kwargs)
    else:
        args = kwargs

    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print_arg_set(**kwargs)
    exproot, expdirs = make_data_inference(
        args['filename'], args['writeto'], args['sub_dirs'], args['tilesize'],
        args['writesize'], args['overlap'], args['remove_first'])

    if args['tileonly']:
        print '\nDone processing {}; Returning\n'.format(args['filename'])
        return

    # Main function here:
    run_histoseg(exproot, expdirs, args['weights'], args['model_template'],
                 args['caffe_mode'], args['GPU_ID'], args['dev'])

    if do_assembly:
        downsample_overlap = get_downsample_overlap(
            args['tilesize'], args['writesize'], args['overlap'])

        assemble_tiles(exproot, expdirs, args['writesize'], downsample_overlap,
                       args['overlay'], args['filename'], args['tilesize'])

    if do_clean:
        cleanup(expdirs)


def print_arg_set(**kwargs):
    print '\n\tArg set:'
    for arg in kwargs:
        print '\t\t{} : {}'.format(arg, kwargs[arg])


##################################################################
##################################################################
###
###       ~~~~~~~~~~~~ Combine scales ~~~~~~~~~~~~
###
##################################################################
##################################################################


def pull_svs_stats(svs):
    # Specficially for svs files
    app_mag = svs.properties['aperio.AppMag']
    nlevels = svs.level_count
    level_dims = svs.level_dimensions

    # Find 20X level:
    if app_mag == '20':  # scanned @ 20X
        return 0, level_dims[0]
    if app_mag == '40':  # scanned @ 40X
        return 1, level_dims[1]

'''
It works with powers of 2, because the division is probably by 2, 4 or 16
For non powers of 2, it's OK if the size is small because the shift is
small
For other cases, it's I guess possible to afterwards, resize the image
to make up for the lost fractional tiles.
Like... you'll have an image wih 59 tiles in one direction, but since
the downsampled tiles give an under-sampling, because you can't take
fractional pixels.
The solution is to track the difference between the ideal reconstruction
and the practical one.
Then at the end do a small resize to force the practical reconstruction
to be the expected ideal size, in order to properly attach the padding.
The problem is pronounced when area >> tilesize
'''


def pad_m(m, tilesize, svsfile):
    # Infer how much of the original was cut given tilesize and dimensions:
    print ''
    print '[Output from : {}]'.format(PrintFrame())
    leveldims = svsfile.level_dimensions
    downsample = svsfile.level_downsamples
    app_mag = svsfile.properties['aperio.AppMag']
    if app_mag == '20':
        lvl20 = 0
    elif app_mag == '40':
        lvl20 = 1

    ### Sketchy fix
    if int(downsample[-1]) == 64:
        low = len(downsample) - 2
    else:
        low = len(downsample) - 1

    tile_top = int(np.round(leveldims[0][0] * tilesize / leveldims[lvl20][0] /
                            np.sqrt(downsample[lvl20])))  # (EQ 1)
    factor = int(downsample[low])  # This conversion isn't important

    print '\tFactor = {}'.format(factor)

    # Tilesize w.r.t. the target size (how big to make things)
    # This one can be fractional.
    # When it is fractional, and [r, c] ~ tilesize, there are problems
    ds_tilesize_float = tile_top / float(factor)
    ds_tilesize = tile_top / factor
    real_c = int(m.shape[0] * ds_tilesize)
    real_r = int(m.shape[1] * ds_tilesize)

    # Now get the padding w/r/t the ideal tile size
    ideal_c = int(m.shape[0] * ds_tilesize_float)
    ideal_r = int(m.shape[1] * ds_tilesize_float)

    padc = leveldims[low][0] - ideal_c
    padr = leveldims[low][1] - ideal_r

    d_x = ideal_c / float(real_c)
    d_y = ideal_r / float(real_r)

    print '\tdx = {}'.format(d_x)
    print '\tdy = {}'.format(d_y)

    return padr, padc, ds_tilesize, d_x, d_y


# so ugly
def assemble_full_slide(scales=[756, 512, 256], **kwargs):

    # Still there will be a little bit of disconcordance. Just a bit.
    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tCombining class probabilities across scales'

    expdirs = kwargs['sub_dirs']
    tail = os.path.basename(kwargs['filename'])
    slide_name, ext = os.path.splitext(tail)
    exproot = os.path.join(kwargs['writeto'], slide_name)
    nclass = kwargs['nclass']

    svsfile = OpenSlide(kwargs['filename'])
    level_dims = svsfile.level_dimensions[-1]
    level_dims = np.array(level_dims)
    level_dims = level_dims[::-1]
    ### Change level_dims to account for the overlap and the padding

    scaleimages = [None] * len(scales)
    for k, s in enumerate(scales):
        ds_overlap = get_downsample_overlap(s, kwargs['writesize'],
                                            kwargs['overlap'])
        print ''
        print ' ######################################################################## '
        print ' ######################################################################## '
        print ''
        print '[Output from : {}]'.format(PrintFrame())
        print '\tGathering scale {} ({} of {})'.format(s, k + 1, len(scales))
        print '\tDownscale overlap: {}'.format(ds_overlap)

        # Pull prob matching the scale
        probdirs = [
            os.path.join(exproot, '{}_{}'.format(d, s)) for d in expdirs
            if 'prob' in d
        ]

        # Get the tilemap
        tilemap = 'data_tilemap_{}.npy'.format(s)
        m = np.load(os.path.join(exproot, tilemap))
        r, c = m.shape
        print '\tUsing tilemap {}'.format(tilemap)
        print '\tm = {}'.format(m.shape)

        # Integers
        projected_area = [r * kwargs['writesize'], c * kwargs['writesize']]
        #projected_ratio = projected_area[0] / float(projected_area[1])
        low_padr, low_padc, writesize, dx, dy = pad_m(m, s, svsfile)

        if int(svsfile.level_downsamples[-1]) == 64:
            low = len(svsfile.level_downsamples) - 2
        else:
            low = len(svsfile.level_downsamples) - 1

        low_level_dimensions = svsfile.level_dimensions[low]
        low_level_dimensions = low_level_dimensions[::-1]
        low_level_ratio = low_level_dimensions[0] / float(
            low_level_dimensions[1])
        ylow, xlow = low_level_dimensions
        adjusted_low_level_dims = [ylow - low_padr, xlow - low_padc]
        adjusted_low_level_ratio = adjusted_low_level_dims[0] / float(
            adjusted_low_level_dims[1])
        downsample_factor = projected_area[0] / float(
            adjusted_low_level_dims[0])
        # downsample_place_size = int(kwargs['writesize'] / downsample_factor)
        scaleimages[k] = [
            cv2.resize(data.build_region(
                region=[0, 0, c, r],
                m=m,
                source_dir=pd,
                place_size=writesize,
                overlap=ds_overlap,
                overlay_dir='',
                exactly=low_level_dimensions), (0,0), fx = dx, fy = dy) for pd in probdirs
        ]

    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tWriting class images at each scale'

    def scale_img_write(c, s, img):
        tstr = os.path.join(exproot, 'whole_c{}_s{}.jpg'.format(c, s))
        print '\t{} shape {}'.format(tstr, img.shape)
        cv2.imwrite(tstr, img)

    print '\tScale images: {}'.format(len(scaleimages))
    print '\tsclaeimages[0]: {}'.format(len(scaleimages[0]))
    # This is the pep way to write a list comp??
    _ = [
        scale_img_write(c, s, scaleimages[x][c])
        for c in range(nclass) for x, s in enumerate(scales)
    ]

    # Got all the images like this:
    # scaleimages = [[prob1_s1, prob2_s1,..], [prob1_s2, prob2_s2,..]]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mean_images = [None] * nclass
    for k in range(nclass):
        print ''
        print '[output from : {}]'.format(PrintFrame())
        print '\tgathering class {} ({} of {})'.format(k, k + 1, nclass)
        classim = [si[k] for si in scaleimages]

        print '\tgot {} images'.format(len(classim))
        for kk, x in enumerate(classim):
            print '\timg {} shape: {}'.format(scales[kk], x.shape)

        # combine them somehow
        classim = np.dstack(classim)
        #classim = cv2.morphologyEx(classim, cv2.MORPH_OPEN, kernel)
        classim = np.mean(classim, axis=2)
        # classim = cv2.morphologyEx(classim, cv2.MORPH_OPEN, kernel)

        mean_images[k] = 'class_{}_comboimg.jpg'.format(k)
        mean_images[k] = os.path.join(exproot, mean_images[k])
        print '\twriting to {}'.format(mean_images[k])
        cv2.imwrite(mean_images[k], classim)

    print ''
    print '[output from : {}]'.format(PrintFrame())
    print '\tcombining everything into final class image'
    comboimage = [cv2.imread(mi)[:, :, 0]
                  for mi in mean_images]  # imread makes d = 3
    print '\tComboimage is {} long'.format(len(comboimage))
    comboimage = np.dstack(comboimage)
    comboclass = np.argmax(comboimage, axis=2)

    print ''
    print '[output from : {}]'.format(PrintFrame())
    print '\tcomboclass: {}, {}'.format(comboclass.shape, np.unique(comboclass))

    colors = generate_color.generate(
        n=nclass, whiteidx=kwargs['whiteidx'], cmap='brg')
    comboclass = histoseg.impose_colors(comboclass, colors)
    comboname = os.path.join(exproot, 'multiscale_class.png')
    print '\tsaving to {}'.format(comboname)
    cv2.imwrite(comboname, comboclass)

    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tcoloring from original rgb: '
    avgscale = int(np.mean(scales))
    # Should be same size & scale as combo image
    rgb = svsfile.read_region(
        location=(0, 0), level=low, size=svsfile.level_dimensions[low])
    rgb = np.array(rgb)[:, :, :3]
    rgb = rgb[:, :, (2, 1, 0)]
    rgb = data.overlay_colors(rgb, comboclass)
    comboname = os.path.join(exproot, 'multiscale_colored.jpg')
    print '\tsaving to {}'.format(comboname)
    cv2.imwrite(comboname, rgb)

    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print ' ######################################################################## '
    print ' ######################################################################## '
    print '\tFinished !!!!'


def cleanup_all(exproot):
    dlist = os.listdir(exproot)
    dlist = [os.path.join(exproot, d) for d in dlist]
    _ = [shutil.rmtree(d) for d in dlist if os.path.isdir(d)]


def run_offsets(**kwargs):
    # offsets should really be some fraction of tilesize
    offsets = [0, 128]

    # Classification loop
    for offset in offsets:
        # Re-parse, I guess
        print ''
        print '[Output from : {}]'.format(PrintFrame())

        args = parse_options(**kwargs)
        # Change some things
        args['offset'] = offset
        args['sub_dirs'] = [
            '{}_{}'.format(subdir, args['offset'])
            for subdir in args['sub_dirs']
        ]
        print_arg_set(**args)
        run_inference(
            do_clean=False, do_parsing=False, do_assembly=False, **args)

    # Assembly function
    if not kwargs['tileonly']:
        print ''
        print '[Output from : {}]'.format(PrintFrame())
        print '\tEntering assembly procedure for {}'.format(kwargs['filename'])
        print_arg_set(**kwargs)
        assemble_full_slide(scales=scales, **kwargs)

        tail = os.path.basename(kwargs['filename'])
        slide_name, ext = os.path.splitext(tail)
        exproot = os.path.join(kwargs['writeto'], slide_name)
        print 'Cleaning up in {}'.format(exproot)
        cleanup_all(exproot)


def run_multiscale(**kwargs):
    # scales = [556, 512, 496, 458]
    scales = [512, 600, 726]

    for s in scales:
        # Re-parse, I guess
        print ''
        print '[Output from : {}]'.format(PrintFrame())

        args = parse_options(**kwargs)
        # Overwrite some settings
        args['tilesize'] = s  # Override tilesize
        args['sub_dirs'] = [
            '{}_{}'.format(subdir, args['tilesize'])
            for subdir in args['sub_dirs']
        ]
        # args['remove_first'] = True
        print_arg_set(**args)
        run_inference(
            do_clean=False, do_parsing=False, do_assembly=False, **args)

    #if not kwargs['tileonly']:
    #    print ''
    #    print '[Output from : {}]'.format(PrintFrame())
    #    print '\tEntering assembly procedure for {}'.format(kwargs['filename'])
    #    print_arg_set(**kwargs)
    #    assemble_full_slide(scales=scales, **kwargs)

    #    tail = os.path.basename(kwargs['filename'])
    #    slide_name, ext = os.path.splitext(tail)
    #    exproot = os.path.join(kwargs['writeto'], slide_name)
    #    print 'Cleaning up in {}'.format(exproot)
    #    cleanup_all(exproot)

    return 0


##################################################################
##################################################################
###
###       ~~~~~~~~~~~~~~~ working functions ~~~~~~~~~~~~~~~~~
###
##################################################################
##################################################################


def run_devel():
    filename = '/home/nathan/data/pca_wsi/swartwoods.svs'
    # filename = '/home/nathan/data/pca_wsi/MaZ-001-a.svs'
    writeto = '/home/nathan/histo-seg/pca'
    tilesize = 512
    writesize = 256  # this remains the dim expected by the network
    overlap = 64
    remove = True

    weights = '/home/nathan/semantic-pca/weights/seg_0.5/norm_iter_125000.caffemodel'
    model_template = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'
    caffe_mode = 0
    GPU_ID = 0
    dev = False

    sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4']

    run_multiscale(
        filename=filename,
        writeto=writeto,
        sub_dirs=sub_dirs,
        tilesize=tilesize,
        writesize=writesize,
        weights=weights,
        model_template=model_template,
        remove_first=remove,
        overlap=overlap,
        dev=dev,
        nclass=5,
        whiteidx=3)


def parse_options(**kwargs):
    print '\tParsing arguments: '

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
        'dev': False,
        'nclass': 5,
        'whiteidx': 0
    }

    for arg in kwargs:
        print '\t\t{} : {}'.format(arg, kwargs[arg])
        #passed_in[arg] = kwargs[arg]

    # Check what is defined, and assign defaults:
    for d in defaults:
        if d in kwargs:
            pass
        else:
            print '\t\tUsing default value for {}'.format(d)
            kwargs[d] = defaults[d]

    if None in kwargs.itervalues():
        raise Exception('All the paths must be set')

    return kwargs


if __name__ == '__main__':
    run_devel()
    '''

// Notes

- Development direction:
    multi-scale classification -
    use a network up-front to focus on cancer/non cancer areas from low resolution
    We want a very sensitive filter, not to miss any of the cancer in other words.

- Scale 0 classification: define the rough tumor area
- Scale 1 segmentation: fine-grained segmentation and grading

'''
