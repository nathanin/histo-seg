'''

Collection of run() functions from data, training and histoseg

'''
import os
import data
import histoseg # Mine
import shutil
import glob

from openslide import OpenSlide
import numpy as np


def run_histoseg(exphome, expdirs, weights, model_template, mode, GPU_ID):
    ## Echo inputs
    print "\nRunning histoseg.process: "
    print "Source: {}".format(expdirs[0])
    print "Destination: {}".format(expdirs[1:])
    print "Model template: {}".format(model_template)
    print "Weights: {}".format(weights)
    print "Mode: {}".format(mode)
    print "Running on GPU: {}".format(GPU_ID)
   
    histoseg.process(exphome, expdirs, model_template, 
                     weights, mode, GPU_ID)


def make_data_inference(filename, writeto, create, tilesize, 
                        writesize, overlap = 0, remove_first = False):
    print '\nRunning data creation for inference:'
    print 'File: {}'.format(filename)
    print 'Destination: {}'.format(writeto)
    return data.make_inference(filename, writeto, create, tilesize, 
                               writesize, overlap, remove_first)



def assemble_tiles(result_root, expdirs, writesize, overlap, overlay,
                   filename, tilesize):
    print "\nAssembling tiles:"
    print "Saving result to : {}".format(result_root)
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
                'tileonly': False}

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
    # print "\nFinal arg set:"
    # for arg in kwargs:
    #     print "{} : {}".format(arg, kwargs[arg])
    # Everything's set; paths aren't allowed to be None:
    if None in kwargs.itervalues():
        raise Exception('All the paths must be set')
    return kwargs 


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
                 args['GPU_ID'])

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
    print "\nArg set:"
    for arg in kwargs:
        print "{} : {}".format(arg, kwargs[arg])


def assemble_full_slide(scales= [756, 512, 256], **kwargs):
    # With N number of scales, average the probability images from each:
    # The catch:
    # We have to work on the whole slide at once, it's the only way to be
    # reasonably sure to align all the scales

    # Still there will be a little bit of disconcordance. Just a bit.
    exproot = kwargs['writeto']
    expdirs = kwargs['sub_dirs']

    svsfile = OpenSlide(kwargs['filename'])
    level_0_dims = svsfile.level_dimensions[2] # common target size; pretty small.

    scaleimages = []
    for k,s in enumerate(scales):
        # Pull prob matching the scale
        probdirs = ['{}_{}'.format(d, s) in expdirs if 'prob' in expdirs]

        # Get the tilemap
        tilemap = 'data_tilemap_{}.npy'.format(s)
        m = np.load(tilemap)
        w,h = m.shape

        # Construct scaled images
        scaleimages[k] = [data.build_region(region = [0,0,w,h], m = m, source_dir = pd,
                                place_size = kwargs['writesize'], overlap = kwargs['overlap'],
                                overlay_dir = '', max_w = level_0_dims[0]) for pd in probdirs]

    # Got all the images like this:
    # scaleimages = [[prob1_s1, prob2_s1,..], [prob1_s2, prob2_s2,..]]
    for k,s in enumerate(scales):
        scale_ = [si[k] for si in scaleimages]

        # Combine them somehow
        scale_ = np.dstack(scale_)
        scale_ = np.mean(scale_, axis = 2)

    

def run_multiscale(**kwargs):
    scales = [756, 512, 256]

    for s in scales:
        # Re-parse, I guess
        print 'Working in scale: {}'.format(s)
        args = parse_options(**kwargs)

        # Remove some things
        args['tilesize'] = s # Override tilesize 
        args['sub_dirs'] = ['{}_{}'.format(subdir, args['tilesize']) for subdir in args['sub_dirs']]
        args['remove_first'] = True
        print_arg_set(**args)
        run_inference(do_clean = False, do_parsing = False, **args)

    # Now all the resolution's and outputs exist
    #assemble_full_slide(scales= scales, **kwargs)


if __name__ == "__main__":
    filename = "/home/nathan/data/pca_wsi/MaZ-001-a.svs"
    writeto = "/home/nathan/histo-seg/pca"
    tilesize = 512
    writesize = 256 # this remains the dim expected by the network
    overlap = 64
    remove = True

    weights = "/home/nathan/semantic-pca/weights/seg_0.4/norm_iter_95000.caffemodel"
    model_template = "/home/nathan/histo-seg/code/segnet_basic_inference.prototxt"
    caffe_mode = 0
    GPU_ID = 0

    sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4']

    run_multiscale(filename = filename,
                   writeto = writeto,
                   sub_dirs = sub_dirs,
                   tilesize = tilesize,
                   weights = weights,
                   model_template = model_template,
                   remove_first = remove,
                   overlap = overlap)
