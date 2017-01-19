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



if __name__ == "__main__":
    filename = "/Users/nathaning/Dropbox/SVS/PCA/MaZ.svs"
    writeto = "/Users/nathaning/histo-seg/pca"
    tilesize = 512
    writesize = 256 # this remains the dim expected by the network
    overlap = tilesize/10
    remove_first = True

    weights = "/Users/nathaning/Dropbox/projects/semantic_pca/weights/pca_segnet_dec7_norm_65000.caffemodel"
    model_template = "/Users/nathaning/histo-seg/code/segnet_basic_inference.prototxt"
    mode = 1 # 0 - GPU, 1 - CPU
    overlay = 1

    tiledir, exproot, created_dirs = make_data_inference(filename, writeto, 
            ["tiles", "result"], 
            tilesize, writesize, 
            overlap, remove_first)

    # created_dirs is the list of dirs we have made and expect to be populated
    # tiledir and exproot are special ... ? So should result be special too? Idk. 
   
    ## We"re left with created_dirs = ['result', 'labels', 'prob']
    #exproot = "/Users/nathaning/histo-seg/pca/MaZ"
    #tiledir = "/Users/nathaning/histo-seg/pca/MaZ/tiles"
    #resultdir = "/Users/nathaning/histo-seg/pca/MaZ/result"

    #created_dirs[0] is result
    #resultdir = created_dirs[0]
    run_histoseg( exproot, tiledir, created_dirs, weights, model_template, mode )
  
    #exproot = '/Users/nathaning/histo-seg/pca/MaZ'

    # For overlapping: translate the overlap to the writesize:
    tilesize = tilesize + 2*overlap # New tilesize, after factoring the ovlp
    downsample_factor = tilesize/writesize 
    ds_overlap = overlap / downsample_factor # Pixels extra that are encoded into each tile

    assemble_tiles( exproot, created_dirs, writesize, ds_overlap, overlay)

    # Clean extra space from the project. 
    cleanup(created_dirs)


