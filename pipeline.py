'''
Collection of run() functions from data, training and histoseg

'''

import data


def make_data_inference(filename, writeto, tilesize, writesize, remove_first = False):
    print "Running data creation for inference:"
    print "File: {}".format(filename)
    print "Destination: {}".format(writeto)
    return data.run_inference(filename, writeto, tilesize, writesize, remove_first)

 



if __name__ == '__main__':
   filename = '/home/nathan/data/pca_wsi/MaZ-001-a.svs'
   writeto = '/home/nathan/histo-seg/pca'
   tilesize = 2048
   writesize = 256
   remove_first = True

   tiledir, exproot = make_data_inference(filename, writeto, tilesize, writesize, remove_first)
    
