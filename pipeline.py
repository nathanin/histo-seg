'''

Collection of run() functions from data, training and histoseg

'''

import data


def make_data_inference(filename, writeto, tilesize, writesize, remove_first = False):
    print "Running data creation for inference:"
    print "File: {}".format(filename)
    print "Destination: {}".format(writeto)
    return data.run_inference(filename, writeto, tilesize, writesize, remove_first)

 
def assemble_tiles(result_root, source_dirs):
    print "Assembling tiles:"

    data.assemble(result_root, source_dirs)

if __name__ == '__main__':
    filename = '/Users/nathaning/Dropbox/SVS/PCA/MaZ.svs'
    writeto = '/Users/nathaning/histo-seg/pca'
    tilesize = 2048
    writesize = 256
    remove_first = True

    #tiledir, exproot = make_data_inference(filename, writeto, tilesize, writesize, remove_first)
  
    exproot = '/Users/nathaning/histo-seg/pca/MaZ'
    assemble_tiles(exproot, ['result'])
