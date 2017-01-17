'''
One-stop solution to data pruning, preprocessing, and conversion in format usable to caffe-segnet

Two modes: training and inference

Training takes two dirs, with images, similarly labeled.
	The source images are multiplied and written out to _prepared folders

Inference takes an whole slide file
	A new dir is created underneath a <project_dir> with the name of the file.
	The image is pre-processed at low-mag for tissue area
	tissue area is tiled (with/without overlapping) and added to a `tiles` dir
	A list.txt is written to <project_dir>/<image_name>/list.txt
	A map.png and map.npy are written to <project_dir>/<image_name>

'''

from openslide import OpenSlide
import cv2
import colorNormalization as cnorm
import numpy as np
import os
import shutil

def pull_svs_stats(svs):
    # Specficially for svs files
    app_mag = svs.properties['aperio.AppMag']
    nlevels = svs.level_count
    level_dims = svs.level_dimensions

    # Omit print statement

    # Find 20X level:
    if app_mag == '20': # scanned @ 20X
        return 0, level_dims[0]
    if app_mag == '40': # scanned @ 40X
        return 1, level_dims[1]


def check_white(tile, cutoff = 225, pct = 0.5):
    # True if tile > cutoff is less than pct
    gray = cv2.cvtColor(tile, cv2.COLOR_RGBA2GRAY)
    white = (gray > cutoff).sum() / float(gray.size)

    return white < pct


def write_tile(tile, filename, writesize):
    # Needed for Inference mode ***
    tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB) # ???
    tile = tile[:,:,(2,1,0)] # ???
    #tile = cv2.resize(tile, dsize = (writesize, writesize)) # Before norm; for speed??
    tile = cnorm.normalize(image = tile, target = None, verbose = False)
    tile = cv2.resize(tile, dsize = (writesize, writesize)) # Before norm; for speed??
   
    cv2.imwrite(filename = filename, img = tile)


def update_map(tilemap, x, y, value):
    # Record a value at position (x,y) in the 2-array tilemap
    tilemap[y,x] = value

    return tilemap

def tile_wsi(wsi, tilesize, writesize, writeto, prefix = 'tile'):
    '''
    Reworked version from tile_wsi.py

    nrow 0) = nrow(lvl)
    y(0) = factor * y(lvl) ---> factor = y(0) / y(lvl)
    tilesize(0) = factor * tilesize(lvl)
    tilesize(0) = y(0) * tilesize(lvl) / y(lvl) (EQ 1)

    Remove logging functions and max_tile functions
    '''

    lvl20x, dim20x = pull_svs_stats(wsi)

    resize_factor = int(wsi.level_downsamples[lvl20x]) # 1 if the slide is 20x
    
    # Ratio the requested tilesize (w.r.t. 20X) back to the top-level in the file. 
    dims_top = wsi.level_dimensions[0]
    tile_top = int(dims_top[0] * tilesize / dim20x[0] / np.sqrt(resize_factor)) # (EQ 1)

    print "tilesize w.r.t. level 0 = {}".format(tile_top)
    print "tilesize w.r.t. 20x level ({}) = {}".format(lvl20x, tilesize)

    # Whole numbers of tiles: (ASSUME tiles are always square ~~~ )
    nrow = dims_top[1] / tile_top
    ncol = dims_top[0] / tile_top
    
    
    # Forego printing slide & tile info TODO add pretty print for logging

    # Construct the coordinate lattice:
    lst = [(k,j) for k in range(nrow) for j in range(ncol)]
    tilemap = np.zeros(shape = (nrow, ncol), dtype = np.uint32)
    ntiles = len(lst) # == nrow * ncol


    # TODO: add chuck read's and low-res whitespace mapping
    print "Populating tilemap"
    print "Created a {} row by {} col lattice over the image".format(nrow, ncol)
    print "Pulling out squares side length = {}".format(tilesize),
    for index, coords in enumerate(lst):
        # Coordinates of tile's upper-left corner w.r.t. the predfined lattice
        [y, x] = coords

        name = '{}{}.jpg'.format(prefix, index)

        tile = wsi.read_region(location = (x*tile_top, y*tile_top), 
                level = lvl20x, 
                size =(tilesize, tilesize)) # size= () may be w.r.t. the target level...
        # tile is a PIL.Image:
        tile = np.array(tile)

        # Decide if the tile is white: If OK, then write it
        if check_white(tile):
            filename = os.path.join(writeto, name)
            write_tile(tile, filename, writesize)

            tilemap = update_map(tilemap, x, y, index)

    # Forego printing functions; TODO add logging

    return tilemap

def flip(t):
    pass # make_data.py

def rotate(t):
    pass # make_data.py

def coloration(t):
    pass # make_data.py

def writeList(t):
    pass # make_data.py

def multiply_data(targets):
    pass # make_data.py

def find_bcg(wsi):
    pass # make_data.py

def training():
    pass # make_data.py

def inference():

    pass


def create_dirs_inference(filename, writeto, sub_dirs = ['tiles', 'result', 'labels', 'prob'], remove = False):
    tail = os.path.basename(filename)
    slide_name, ex = os.path.splitext(tail)
    result_root = os.path.join(writeto, slide_name)

    if remove and os.path.exists(result_root):
        shutil.rmtree(result_root)
        os.makedirs(result_root)

    create_dirs = [os.path.join(result_root, d) for d in sub_dirs]
    if not os.path.exists(create_dirs[0]):
        _ = [os.makedirs(d) for d in create_dirs]

    create_dirs.append(result_root)

    return create_dirs
    

def run_inference(filename, writeto, tilesize = 512, writesize = 256, remove_first = False):
    created_dirs = create_dirs_inference(filename, writeto, remove = remove_first) 

    tiledir , resultdir, labeldir, probdir, result_root = created_dirs 
    print "Created dirs: {}".format(created_dirs)
    #_ = [print "{}".format(d) for d in created_dirs]

    wsi = OpenSlide(filename)
    tilemap = tile_wsi(wsi, tilesize, writesize, tiledir)

    # Write out map file as npy
    map_file = os.path.join(result_root, 'data_tilemap.npy')
    np.save(file = map_file, arr = tilemap)

    wsi.close()

    return tiledir, result_root



