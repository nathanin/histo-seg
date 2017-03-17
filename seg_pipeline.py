'''

Collection of run() functions from data, training and histoseg

'''
import os
import data
import histoseg
import shutil


def run_histoseg(exphome, source, dest, weights, model_template, mode, GPU_ID):
    ## Echo inputs
    print '\nRunning histoseg.process: '
    print 'Source: {}'.format(source)
    print 'Destination: {}'.format(dest)
    print 'Model template: {}'.format(model_template)
    print 'Weights: {}'.format(weights)
    print 'Mode: {}'.format(mode)
    print 'Running on GPU: {}'.format(GPU_ID)
   
    histoseg.process(exphome, source, dest, model_template, 
                     weights, mode, GPU_ID)


def make_data_inference(filename, writeto, create, tilesize, 
                        writesize, overlap = 0, remove_first = False):
    print '\nRunning data creation for inference:'
    print 'File: {}'.format(filename)
    print 'Destination: {}'.format(writeto)
    return data.make_inference(filename, writeto, create, tilesize, 
                               writesize, overlap, remove_first)

 
def assemble_tiles(result_root, source_dirs, writesize, overlap, overlay):
    print '\nAssembling tiles:'
    print 'Saving result to : {}'.format(result_root)
    data.assemble(result_root, source_dirs, writesize, overlap, overlay)

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

    print '\nFinal arg set:'
    for arg in kwargs:
        print '{} : {}'.format(arg, kwargs[arg])
    
    # Everything's set; paths aren't allowed to be None:
    if None in kwargs.itervalues():
        raise Exception('All the paths must be set')
    return kwargs 

def run_inference(**kwargs):
    args = parse_options(**kwargs)

    tiledir, exproot, created = make_data_inference(args['filename'],
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
                 tiledir,
                 created,
                 args['weights'],
                 args['model_template'],
                 args['caffe_mode'],
                 args['GPU_ID'])

    downsample_overlap = get_downsample_overlap(args['tilesize'],
                                                args['writesize'],
                                                args['overlap'])

    assemble_tiles(exproot,
                   created,
                   args['writesize'], 
                   downsample_overlap, 
                   args['overlay'] )

    cleanup(created)



if __name__ == '__main__':
    filename = '/Users/nathaning/Dropbox/SVS/PCA/MaZ-001-a.svs'
    writeto = '/Users/nathaning/histo-seg/pca'
    tilesize = 512
    writesize = 256 # this remains the dim expected by the network
    overlap = 64
    remove_first = True

    weights = '/Users/nathaning/Dropbox/projects/semantic_pca/weights/pca_segnet_dec7_norm_65000.caffemodel'
    model_template = '/Users/nathaning/histo-seg/code/segnet_basic_inference.prototxt'
    mode = 0 # 0 - GPU, 1 - CPU
    overlay = 1
    GPU_ID = 0

    print ''
    tiledir, exproot, created_dirs = make_data_inference(filename, writeto, 
            ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3'], 
            tilesize, writesize, 
            overlap, remove_first)

    print ''
    print 'Experiment root: {}'.format(exproot) 
    print ''
    #print 'Tiles: {}'.format(tiledir)
    #print 'Outputs: {}'.format(created_dirs)
    
    #exproot = '/home/nathan/histo-seg/pca/MaZ-001'
    #tiledir = '/home/nathan/histo-seg/pca/MaZ-001/tiles'
    #created_dirs = ['/home/nathan/histo-seg/pca/MaZ-001/result',
    #                '/home/nathan/histo-seg/pca/MaZ-001/prob']
    run_histoseg( exproot, tiledir, created_dirs, weights, model_template, mode, GPU_ID)
  
    # For overlapping: translate the overlap to the writesize:
    # Pixels extra that are encoded into each tile
    ds_overlap = get_downsample_overlap(tilesize, writesize, overlap) 

    assemble_tiles( exproot, created_dirs, writesize, ds_overlap, overlay )

    # Clean extra space from the project. 
    #cleanup(created_dirs)
