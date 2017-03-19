'''

Collection of run() functions from data, training and histoseg

'''
import os
import data
import histoseg # Mine
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
    ## Echo inputs
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
        print 'RUNNING DEVELOPMENT PROCES'
        histoseg.process_dev(expdirs)
    else:
        histoseg.process(exphome, expdirs, model_template, 
                         weights, mode, GPU_ID)


def make_data_inference(filename, writeto, create, tilesize, 
                        writesize, overlap = 0, remove_first = False):
    print '[Output from : {}]'.format(PrintFrame()) 
    print '\tRunning data creation for inference:'
    print '\tFile: {}'.format(filename)
    print '\tDestination: {}'.format(writeto)
    return data.make_inference(filename, writeto, create, tilesize, 
                               writesize, overlap, remove_first)


def assemble_tiles(result_root, expdirs, writesize, overlap, overlay,
                   filename, tilesize):
    print '[Output from : {}]'.format(PrintFrame()) 
    print '\tAssembling tiles:'
    print '\tSaving result to : {}'.format(result_root)
    area_cutoff = data.calc_tile_cutoff(filename, tilesize)
    data.assemble(result_root, expdirs, writesize, overlap, 
                  overlay, area_cutoff, tilesize)


def cleanup(dirs):
    for d in dirs:
        print 'Cleaning {}'.format(d)
        shutil.rmtree(d)


def get_downsample_overlap(tilesize, writesize, overlap):
    ts = tilesize + 2*overlap
    factor = ts / float(writesize)
    
    return int(overlap / factor)


def run_inference(do_clean = True, do_parsing = True, **kwargs):
    if do_parsing:
        args = parse_options(**kwargs)
    else:
        args = kwargs
    exproot, expdirs = make_data_inference(args['filename'],
                                           args['writeto'],
                                           args['sub_dirs'],
                                           args['tilesize'],
                                           args['writesize'],
                                           args['overlap'],
                                           args['remove_first'])

    if args['tileonly']:
        print 'Done processing {}; Returning'.format(args['filename'])
        return

    run_histoseg(exproot, 
                 expdirs,
                 args['weights'],
                 args['model_template'],
                 args['caffe_mode'],
                 args['GPU_ID'],
                 args['dev'])

    downsample_overlap = get_downsample_overlap(args['tilesize'],
                                                args['writesize'],
                                                args['overlap'])

    assemble_tiles(exproot,
                   expdirs,
                   args['writesize'], 
                   downsample_overlap, 
                   args['overlay'],
                   args['filename'],
                   args['tilesize'])

    if do_clean:
        cleanup(created)


def print_arg_set(**kwargs):
    print '\nArg set:'
    for arg in kwargs:
        print '{} : {}'.format(arg, kwargs[arg])

##################################################################
##################################################################
###
###       ~~~~~~~~~~~~ Combine scales ~~~~~~~~~~~~
###
##################################################################
##################################################################


def pad_m(m, tilesize, svsfile):
    # Infer how much of the original was cut given tilesie and dimensions:
    print ''
    print '[Output from : {}]'.format(PrintFrame())
    leveldims = svsfile.level_dimensions
    downsample = svsfile.level_downsamples
    app_mag = svsfile.properties['aperio.AppMag']
    if app_mag == '20':
        lvl20 = 0
    elif app_mag == '40':
        lvl20 = 1

    tile_top = int(leveldims[0][0] * tilesize / leveldims[lvl20][0] / np.sqrt(downsample[lvl20])) # (EQ 1)
    r,c = m.shape

    r0 = r*tile_top
    c0 = c*tile_top

    c_, r_ = leveldims[0] # Flipped from the OpenCV convention
    # This reversal is the source of great confusion for me. 

    print '\tTile top {}'.format(tile_top)
    print '\tM is {}'.format(m.shape)
    print '\tOriginal dims: {} , tiles cover: {}, downsample {}'.format(
                                        (r_, c_), (r0, c0), downsample)
    rpad = int((r_ - r0) / np.sqrt(downsample[-1]))
    cpad = int((c_ - c0) / np.sqrt(downsample[-1]))

    print '\tPadding w.r.t. last level is rows {}, cols {}'.format(rpad, cpad) 
    return rpad, cpad 

# so ugly
def assemble_full_slide(scales= [756, 512, 256], **kwargs):
    # With N number of scales, average the probability images from each:
    # The catch:
    # We have to work on the whole slide at once, it's the only way to be
    # reasonably sure to align all the scales

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

    scaleimages = [None]*len(scales)
    for k,s in enumerate(scales):
        ds_overlap = get_downsample_overlap(s, kwargs['writesize'], kwargs['overlap'])
        print ''
        print '[Output from : {}]'.format(PrintFrame())
        print '\tGathering scale {} ({} of {})'.format(s, k+1, len(scales))
        print '\tDownscale overlap: {}'.format(ds_overlap)

        # Pull prob matching the scale
        probdirs = [os.path.join(exproot, '{}_{}'.format(d, s))
                    for d in expdirs if 'prob' in d]

        # Get the tilemap
        tilemap = 'data_tilemap_{}.npy'.format(s)
        print '\tUsing tilemap {}'.format(tilemap)
        m = np.load(os.path.join(exproot, tilemap))
        h,w = m.shape

        # Construct scaled images
        scaleimages[k] = [data.build_region(region = [0,0,w,h], m = m, source_dir = pd,
                            place_size = kwargs['writesize'], overlap = ds_overlap,
                            overlay_dir = '', exactly = level_dims, 
                            pad = pad_m(m, s, svsfile)) for pd in probdirs]
        print '\tScale {} image: {}'.format(s, scaleimages[k][0].shape)


    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tWriting class images at each scale'
    def scale_img_write(c, s, img):
        tstr = os.path.join(exproot, 'whole_c{}_s{}.jpg'.format(c, s))
        print '\t{} shape {}'.format(tstr, img.shape)
        cv2.imwrite(tstr, img)

    print '\tScale images: {}'.format(len(scaleimages))
    print '\tsclaeimages[0]: {}'.format(len(scaleimages[0]))
    _ = [scale_img_write(c, s, scaleimages[x][c]) 
         for c in range(nclass) 
         for x,s in enumerate(scales)]

    # Got all the images like this:
    # scaleimages = [[prob1_s1, prob2_s1,..], [prob1_s2, prob2_s2,..]]
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    # mean_images = [None]*nclass
    # for k in range(nclass):
    #     print ''
    #     print '[Output from : {}]'.format(PrintFrame())
    #     print '\tGathering class {} ({} of {})'.format(k, k+1, nclass)
    #     classim = [si[k] for si in scaleimages]

    #     print '\tGot {} images'.format(len(classim))
    #     for kk,x in enumerate(classim): print '\timg {} shape: {}'.format(scales[kk], x.shape)

    #     # Combine them somehow
    #     classim = np.dstack(classim)
    #     #classim = cv2.morphologyEx(classim, cv2.MORPH_OPEN, kernel)
    #     classim = np.mean(classim, axis = 2)
    #     classim = cv2.morphologyEx(classim, cv2.MORPH_OPEN, kernel)

    #     mean_images[k] = 'class_{}_comboimg.jpg'.format(k)
    #     mean_images[k] = os.path.join(exproot, mean_images[k])
    #     print '\tWriting to {}'.format(mean_images[k])
    #     cv2.imwrite(mean_images[k], classim)


    # print 'Combining everything into final class image'
    # print ''
    # print '[Output from : {}]'.format(PrintFrame())
    # comboimage = [cv2.imread(mi)[:,:,0] for mi in mean_images]
    # comboimage = np.dstack(comboimage)
    # comboclass = np.argmax(comboimage, axis = 2)

    # print ''
    # print '[Output from : {}]'.format(PrintFrame())
    # print '\tcomboclass: {}, {}'.format(comboclass.shape, np.unique(comboclass))

    # colors = generate_color.generate(n = nclass)
    # comboclass = histoseg.impose_colors(comboclass, colors)
    # comboname = os.path.join(exproot, 'multiscale_class.jpg')
    # print '\tSaving to {}'.format(comboname) 
    # cv2.imwrite(comboname, comboclass)

    # print ''
    # print '[Output from : {}]'.format(PrintFrame())
    # print '\tColoring from original rgb: '
    # # Width and height somehow reverse here. 
    # rgb = svsfile.read_region(location = (0,0),
    #                           level = len(svsfile.level_dimensions)-1, 
    #                           size = svsfile.level_dimensions[-1])
    # rgb = np.array(rgb)[:,:,:3]
    # rgb = data.overlay_colors(rgb, comboclass)
    # comboname = os.path.join(exproot, 'multiscale_colored.jpg')
    # print '\tSaving to {}'.format(comboname) 
    # cv2.imwrite(comboname, rgb)

    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print 'Made it'



def run_multiscale(**kwargs):
    # scales = [556, 512, 496, 458]
    scales = [798, 756, 512]

    # for s in scales:
    #     # Re-parse, I guess
    #     print 'Working in scale: {}'.format(s)
    #     args = parse_options(**kwargs)
    #     # Remove some things
    #     args['tilesize'] = s # Override tilesize 
    #     args['sub_dirs'] = ['{}_{}'.format(subdir, args['tilesize']) 
    #                         for subdir in args['sub_dirs']]
    #     args['remove_first'] = True
    #     print_arg_set(**args)
    #     run_inference(do_clean = False, do_parsing = False, **args)

    assemble_full_slide(scales= scales, **kwargs)



##################################################################
##################################################################
###
###       ~~~~~~~~~~~~~~~ working functions ~~~~~~~~~~~~~~~~~
###
##################################################################
##################################################################


def dev_mode():
    # filename = '/home/nathan/data/pca_wsi/MaZ-001-a.svs'
    # writeto = '/home/nathan/histo-seg/pca'

    filename = '/Users/nathaning/Dropbox/SVS/PCA/MaZ-001-a.svs'
    writeto = '/Users/nathaning/_projects/histo-seg/pca'

    tilesize = 512
    writesize = 256 # this remains the dim expected by the network
    overlap = 64
    remove = True

    weights = '/home/nathan/semantic-pca/weights/seg_0.5/norm_iter_125000.caffemodel'
    model_template = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'
    caffe_mode = 0
    GPU_ID = 0
    dev = True

    # sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4']
    sub_dirs = ['tiles', 'result', 'prob0', 'prob1']

    run_multiscale(filename = filename,
                   writeto = writeto,
                   sub_dirs = sub_dirs,
                   tilesize = tilesize,
                   writesize = writesize,
                   weights = weights,
                   model_template = model_template,
                   remove_first = remove,
                   overlap = overlap,
                   dev = dev,
                   nclass = 2,
                   whiteidx = 0)



def run_mode():
    filename = '/home/nathan/data/pca_wsi/MaZ-001-a.svs'
    writeto = '/home/nathan/histo-seg/pca'
    tilesize = 512
    writesize = 256 # this remains the dim expected by the network
    overlap = 64
    remove = True

    weights = '/home/nathan/semantic-pca/weights/seg_0.5/norm_iter_125000.caffemodel'
    model_template = '/home/nathan/histo-seg/code/segnet_basic_inference.prototxt'
    caffe_mode = 0
    GPU_ID = 0
    dev = False

    sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4']

    run_multiscale(filename = filename,
                   writeto = writeto,
                   sub_dirs = sub_dirs,
                   tilesize = tilesize,
                   writesize = writesize,
                   weights = weights,
                   model_template = model_template,
                   remove_first = remove,
                   overlap = overlap,
                   dev = dev,
                   nclass = 5,
                   whiteidx = 0)


def parse_options(**kwargs):
    print 'Parsing arguments: '

    defaults = {'filename': None,
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
                'whiteidx': 0}

    for arg in kwargs:
        print '{} : {}'.format(arg, kwargs[arg])
        #passed_in[arg] = kwargs[arg]

    # Check what is defined, and assign defaults:
    for d in defaults:
        if d in kwargs:
            pass
        else:
            print 'Using default value for {}'.format(d)
            kwargs[d] = defaults[d]

    if None in kwargs.itervalues():
        raise Exception('All the paths must be set')
    return kwargs 


if __name__ == '__main__':
    dev_mode()
    # run_mode()