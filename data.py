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
import inspect
import glob

def PrintFrame():
    callerframerecord = inspect.stack()[1] #0 represents this line
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    thisfile = info.filename
    thisfun = info.function
    thisline = info.lineno

    return '{} in {} @ {}'.format(thisfile, thisfun, thisline)

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


def write_tile(tile, filename, writesize, normalize):
    # Needed for inference by SegNet
    tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB) # ???
    tile = tile[:,:,(2,1,0)] # ???

    #tile = cv2.resize(tile, dsize = (writesize, writesize)) # Before norm; for speed??
    if normalize:
        tile = cnorm.normalize(image = tile, target = None, verbose = False)
    tile = cv2.resize(tile, dsize = (writesize, writesize)) # Before norm; for speed??
   
    cv2.imwrite(filename = filename, img = tile)


def update_map(tilemap, x, y, value):
    # Record a value at position (x,y) in the 2-array tilemap
    tilemap[y,x] = value

    return tilemap



'''
REVISED PSEUDOCODE 
TODO: IMPLEMENT THIS

it's the opposite of the assembly code.

1. decide how many rows and columns
    - intialize a counting variable
    - initialize a map variable
2. for each row:
    1. load the row
    2. move along the row one tilesize - determine if it passes a check
    3. if it passes the check, write it, increment the counting variable
    4. Record the position of the passing tile in the map variable
    5. continue until the whole number of tiles is exhausted

Then, do some A/B tests to see which one's faster !!!!
The point is to minimize the number of calls to OpenSlide.read_region()
The theory is that read_region() is relatively slow when invoked in serial
and that it's performance is overall faster in reading large chunks at once.
Another piece of assumption is indexing into the np array is faster than read_region()

Maybe there's a performance curve depending on the tile size, or just the number of reads
which is a function of tile size, and the total image size. IDK.

Maybe it doesn't matter. It's just bothering me. 
'''

def tile_wsi(wsi, tilesize, writesize, writeto, overlap = 0, prefix = 'tile'):
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
    overlap20x = int(overlap / np.sqrt(resize_factor))

    print "tilesize w.r.t. level 0 = {}".format(tile_top)
    print "tilesize w.r.t. 20x (level {}) = {}".format(lvl20x, tilesize)
    print "Overlap value w.r.t. level 0 = {}".format(overlap)
    print "Overlap value w.r.t. 20x (level {}) = {}".format(lvl20x, overlap20x)

    # Whole numbers of tiles: (ASSUME tiles are always square ~~~ )
    nrow = dims_top[1] / tile_top
    ncol = dims_top[0] / tile_top
    
    # Forego printing slide & tile info 
    # TODO add pretty print for logging

    # Construct the coordinate lattice:
    # For overlapping: 1 to X-1: pseudo-shrink the available space 
    lst = [(k,j) for k in range(1, nrow-1) for j in range(1, ncol-1)]
    tilemap = np.zeros(shape = (nrow, ncol), dtype = np.uint32)
    ntiles = len(lst) # == nrow * ncol


    # TODO: add chuck read's and low-res whitespace mapping
    print "Populating tilemap"
    print "Created a {} row by {} col lattice over the image".format(nrow, ncol)
    print "Pulling out squares side length = {}".format(tilesize)
    written = 0
    for index, coords in enumerate(lst):
        if index % 100 == 0:
            print "{:05d} / {:05d} ({} written so far)".format(index, ntiles, written)
        # Coordinates of tile's upper-left corner w.r.t. the predfined lattice
        [y, x] = coords

        name = '{}{}.jpg'.format(prefix, index)

        # OVERLAP OPTION #1: Change here the values for tilesize
        tile = wsi.read_region(location = (x*tile_top - overlap, y*tile_top - overlap), 
                level = lvl20x, 
                size =(tilesize + 2*overlap, tilesize + 2*overlap)) # size= () may be w.r.t. the target level...
        # tile is a PIL.Image:
        tile = np.array(tile)

        # OVERLAP OPTION #2: Here, just pad the tile before shrinking

        # Decide if the tile is white: If OK, then write it
        if check_white(tile):
            filename = os.path.join(writeto, name)
            write_tile(tile, filename, writesize, True)

            tilemap = update_map(tilemap, x, y, index)
            written += 1

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

def make_training():
    pass # make_data.py

def create_dirs_inference(filename, writeto, sub_dirs, remove = False):
    tail = os.path.basename(filename)
    slide_name, ex = os.path.splitext(tail)
    exp_home = os.path.join(writeto, slide_name)

    if remove and os.path.exists(exp_home):
        shutil.rmtree(exp_home)
        os.makedirs(exp_home)

    created_dirs = [os.path.join(exp_home, d) for d in sub_dirs]
    if not os.path.exists(created_dirs[0]):
        _ = [os.makedirs(d) for d in created_dirs]


    return exp_home, created_dirs


# New: adding overlap option
'''
Method for overlapping:
    - Create lattice without considering overlap
    - Add overlap to tilesize in both dims
    - Writesize remains 256; that's what the network wants. (change this by not being dum)
'''
def make_inference(filename, writeto, create, tilesize = 512, writesize = 256, overlap = 0, remove_first = False):
    
    exp_home, created_dirs = create_dirs_inference(filename, writeto, sub_dirs = create, remove = remove_first) 
    tiledir = created_dirs[0]
    for d in created_dirs:
        print "Created: {}".format(d)

    wsi = OpenSlide(filename)
    print "Working with slide {}".format(filename)
    
    # Echo the settings:
    print "Settings _________"
    print "Tilesize: {}".format(tilesize)
    print "Write size: {}".format(writesize)
    print "Overlap: {}".format(overlap)
    tilemap = tile_wsi(wsi, tilesize, writesize, tiledir, overlap, prefix = 'tile')

    # Write out map file as npy
    map_file = os.path.join(exp_home, 'data_tilemap.npy')
    np.save(file = map_file, arr = tilemap)

    wsi.close()

    #listfile = write_list_densedata(tiledir, exp_home)

    return tiledir, exp_home, created_dirs[1:]



'''
~~~~~~~~~~~~~~~~~~~~~~~~ ASSEMBLE TILES ~~~~~~~~~~~~~~~~~~~~~~~~
'''

def label_regions(m):
    h,w = m.shape
    bw = m > 0
    bw.dtype = np.uint8 # can't keep bool

    contours, _ = cv2.findContours(image = bw, mode = cv2.RETR_EXTERNAL, 
            method = cv2.CHAIN_APPROX_SIMPLE)
    ll = np.zeros(shape = (h,w), dtype = np.uint8)
    for idx, ct in enumerate(contours):
        cv2.drawContours(ll, [ct], 0, idx+1, thickness = -1) # thickness -1 means to fill

    return ll, contours
    

def get_all_regions(m, threshold = 40):
    '''
    Spit out regions as bounding rects of connected components
    threshold is the number of tiles required (should be careful of large tilesizes)
    '''
    regions = []

    ll, contours = label_regions(m)

    for idx, ct in enumerate(contours):
        if (ll == idx+1).sum() >= threshold:
            regions.append(cv2.boundingRect(ct))

    return regions


def partition_rows(m):
    rows = []
    n_rows = m.shape[0]
    for ix in range(n_rows):
        rows.append(m[ix, :])

    return rows

def empty_block(place_size):
    return np.zeros(shape = (place_size, place_size, 3), dtype = np.uint8)


def load_block(pth, place_size, overlap, interp = cv2.INTER_NEAREST):
    # process overlapping borders...
    # 1. Upsample block to be place_size + 2*overlap
    # 2. Cut out the middle bit
    #print "Block: {}".format(pth),
    
    upsample = place_size + 2*overlap
    #print "Upsampling to: {}".format(upsample) 
    block = cv2.imread(pth)
    block = cv2.resize(block, dsize = (upsample, upsample), interpolation = interp)

    return block[overlap : overlap + place_size, overlap : overlap + place_size, :]


def overlay_colors(img, block):
    img = np.add(img*0.4, block*0.6)
    img = cv2.convertScaleAbs(img)

    return img


def build_row(row, source_dir, place_size,  overlap, overlay_dir = ''):
    do_overlay = os.path.exists(overlay_dir)

    #TODO See if we can do it without initializing with empty col on the left :
    row_out = empty_block(place_size)
    for r in row:
        if r == 0:
            block = empty_block(place_size)
        else:
            # TODO fix the hard-coded "tile" prefix and extension suffix: 
            pth = os.path.join(source_dir, 'tile{}.png'.format(r))

            block = load_block(pth, place_size, overlap)

            if block is None:
                print PrintFrame(),
                print "img {} ought to exist but I guess doesn't.".format(pth)
                block = empty_block(place_size)

            if do_overlay:
            # TODO fix the hard-coded "tile" prefix and extension suffix: 
                ov_pth = os.path.join(overlay_dir, 'tile{}.jpg'.format(r))
                ov_img = load_block(ov_pth, place_size, overlap)
                ov_img = cv2.resize(ov_img, dsize = (place_size, place_size))
                block = overlay_colors(ov_img, block)

        row_out = np.append(row_out, block, axis=1) # Sketchhyyyy

    return row_out
    
def assemble_rows(rows):
    img = rows[0]
    for r in rows[1:]:
        img = np.append(img, r, axis=0)
    return img


def downsize_keep_ratio(img, target_w = 1024, interp = cv2.INTER_NEAREST):
    factor = img.shape[1]/float(target_w)
    target_h = int(img.shape[0] / factor)

    img_new = cv2.resize(src = img, dsize = (target_w, target_h), interpolation = interp)

    return img_new


def build_region(region, m, source_dir, place_size, overlap, overlay_dir, max_w = 10000):
    x,y,w,h = region
    
    print PrintFrame(),
    print " x: {} y: {} w: {} h: {}".format(x,y,w,h)
    # Check if the output will be large
    if w*h*(place_size**2) > (2**31)/3:
        # edit place_size so that the output will fit:
        print PrintFrame(),
        print "Found region > 2**31, resizing to ",
        place_size = int(np.sqrt(((2**31)/3) / (w*h)))
        print "{}".format(place_size)

    rows = partition_rows(m[y:y+h, x:x+w])
    
    built_img = [] # Not really an image; a list of row images
    for ix, r in enumerate(rows):
        #print PrintFrame(),"Row {} / {}".format(ix, len(rows))
        row = build_row(r, source_dir, place_size, overlap, overlay_dir)
        built_img.append(row)

    img = assemble_rows(built_img)

    # Resize down to something sane for writing:
    if img.shape[1] > max_w:
        img = downsize_keep_ratio(img, max_w)
        
    return img


# TODO this function isn't very good. the place to generalize isn't really obvious to me. 
def assemble(exp_home, sources, writesize, overlap, overlay):
    
    # Force sources to be a list:
    if isinstance(sources, basestring):
        sources = [sources]

    # Pull in the image map:
    map_file = os.path.join(exp_home, 'data_tilemap.npy')
    m = np.load(map_file)
    [N,M] = m.shape # Forego printing

    # Pull out disconnected regions that pass a size cutoff:
    regions = get_all_regions(m, threshold = 5)

    overlay_dir = ''
    if overlay:
        # Tiles overlaid onto tiles should work fine.
        overlay_dir = os.path.join(exp_home, 'tiles')

    for index, reg in enumerate(regions):
        # TODO this loop still sucks
        #print PrintFrame(),"Processing region {}".format(index)
        for src in sources:
            reg_name = '{}_{:03d}.jpg'.format(src, index) 
            source_dir = os.path.join(exp_home, src)
            
            print "Pulling images from {}".format(source_dir)
            print "Overlaying onto {}".format(overlay_dir)

            img = build_region(reg, m, source_dir, writesize, overlap, overlay_dir) 
            
            reg_name = os.path.join(exp_home, reg_name)
            cv2.imwrite( reg_name, img )


