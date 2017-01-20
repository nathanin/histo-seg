'''

Collection of run() functions from data, training and histoseg

'''
import os
import data
import histoseg
import shutil


def run_histoseg(exphome, source, dest, weights, model_template, mode):
    ## Echo inputs
    print "Running histoseg.process: "
    print "Source: {}".format(source)
    print "Destination: {}".format(dest)
    print "Model template: {}".format(model_template)
    print "Weights: {}".format(weights)
    print "Mode: {}".format(mode)
   
    histoseg.process(exphome, source, dest, model_template, weights, mode)


def make_data_inference(filename, writeto, create, tilesize, writesize, overlap = 0, remove_first = False):
    print "Running data creation for inference:"
    print "File: {}".format(filename)
    print "Destination: {}".format(writeto)
    return data.make_inference(filename, writeto, create, tilesize, writesize, overlap, remove_first)

 
def assemble_tiles(result_root, source_dirs, writesize, overlap, overlay):
    print "Assembling tiles:"

    data.assemble(result_root, source_dirs, writesize, overlap, overlay)

def cleanup(dirs):
    for d in dirs:
        print "Cleaning {}".format(d)
        shutil.rmtree(d)


def get_downsample_overlap(tilesize, writesize, overlap):
    ts = tilesize + 2*overlap
    factor = tilesize / float(writesize)
    
    return int(overlap / factor)


if __name__ == "__main__":
    filename = "/home/nathan/data/local_ccrcc/08_40X.svs"
    writeto = "/home/nathan/histo-seg/ccrcc"
    tilesize = 256
    writesize = 256 # this remains the dim expected by the network
    overlap = tilesize/8
    remove_first = True

    weights = "/home/nathan/semantic-ccrcc/models/ec2/normalized_ccrcc_dec17_iter_26000.caffemodel"
    model_template = "/home/nathan/histo-seg/code/segnet_basic_inference.prototxt"
    mode = 0 # 0 - GPU, 1 - CPU
    overlay = 1
    GPU_ID = 0

    print ""
    tiledir, exproot, created_dirs = make_data_inference(filename, writeto, 
            ["tiles", "result", "prob0", "prob1", "prob2", "prob3"], 
            tilesize, writesize, 
            overlap, remove_first)

    print ""
    print "Experiment root: {}".format(exproot) 
    print "Tiles: {}".format(tiledir)
    print "Outputs: {}".format(created_dirs)
    
    #exproot = '/home/nathan/histo-seg/pca/MaZ-001'
    #tiledir = '/home/nathan/histo-seg/pca/MaZ-001/tiles'
    #created_dirs = ['/home/nathan/histo-seg/pca/MaZ-001/result',
    #                '/home/nathan/histo-seg/pca/MaZ-001/prob']
    run_histoseg( exproot, tiledir, created_dirs, weights, model_template, mode )
  
    # For overlapping: translate the overlap to the writesize:
    # Pixels extra that are encoded into each tile
    ds_overlap = get_downsample_overlap(tilesize, writesize, overlap) 

    assemble_tiles( exproot, created_dirs, writesize, ds_overlap, overlay )

    # Clean extra space from the project. 
    cleanup(created_dirs)


