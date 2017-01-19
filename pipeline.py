'''

Collection of run() functions from data, training and histoseg

'''

import data
import histoseg

def run_histoseg(exphome, source, dest, weights):
    model_template = os.path.join(exphome, "segnet_basic_inference.prototxt")
    mode = 1
    
    print "Running histoseg.process: "
    print "Source: {}".format(source)
    print "Destination: {}".format(dest)
    print "Model template: {}".format(model_template)
    print "Weights: {}".format(weights)
    print "Mode: {}".format(mode)
   
    histoseg.process(exphome, source, dest, model_template, weights, mode)

def make_data_inference(filename, writeto, tilesize, writesize, overlap = 0, remove_first = False):
    print "Running data creation for inference:"
    print "File: {}".format(filename)
    print "Destination: {}".format(writeto)
    return data.make_inference(filename, writeto, tilesize, writesize, overlap, remove_first)

 
def assemble_tiles(result_root, source_dirs, writesize, overlap):
    print "Assembling tiles:"

    data.assemble(result_root, source_dirs, writesize, overlap)

if __name__ == '__main__':
    filename = '/Users/nathaning/Dropbox/SVS/PCA/MaZ.svs'
    writeto = '/Users/nathaning/histo-seg/pca'
    tilesize = 1024
    writesize = 256 # this remains the dim expected by the network
    overlap = tilesize/10
    remove_first = True

    tiledir, exproot, created_dirs = make_data_inference(filename, 
            writeto, tilesize, writesize, 
            overlap, remove_first)

    # created_dirs is the list of dirs we have made and expect to be populated
    # first and last are: tiledir, and exproot; strip them
    created_dirs = created_dirs[1:-1]

    # We're left with created_dirs = ['result', 'labels', 'prob']
    #run_histoseg should use the contents of created_dirs to decide what outputs to give
    #exproot = '/Users/nathaning/histo-seg/pca/MaZ'
    #tiledir = '/Users/nathaning/histo-seg/pca/MaZ/tiles'
    #writeto = '/Users/nathaning/histo-seg/pca/MaZ/result'
    run_histoseg( tiledir, [created_dirs[0]] )
  
    #exproot = '/Users/nathaning/histo-seg/pca/MaZ'

    # For overlapping: translate the overlap to the writesize:
    tilesize = tilesize + 2*overlap # New tilesize, after factoring the ovlp
    downsample_factor = tilesize/writesize 
    ds_overlap = overlap / downsample_factor # Pixels extra that are encoded into each tile

    assemble_tiles( exproot, [created_dirs[0]] , writesize, ds_overlap)


