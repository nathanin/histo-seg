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

'''
```````````````` DEBUGGING FUNCTOIN ``````````````````
'''
def PrintFrame():
    callerframerecord = inspect.stack()[1] #0 represents this line
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    thisfile = info.filename
    thisfun = info.function
    thisline = info.lineno

    return '{} in {} @ {}'.format(thisfile, thisfun, thisline)
'''
```````````````` DEBUGGING FUNCTOIN ``````````````````
'''
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


def check_white(tile, cutoff = 215, pct = 0.75):
    # True if the tile is mostly non-white
    # Lower pct ~ more strict
    gray = cv2.cvtColor(tile, cv2.COLOR_RGBA2GRAY)
    white = (gray > cutoff).sum() / float(gray.size)

    return white < pct


def write_tile(tile, filename, writesize, normalize):
    # Needed for inference by SegNet
    tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB) # ???
    tile = tile[:,:,(2,1,0)] # ???

    tile = cv2.resize(tile, dsize = (writesize, writesize), 
                      interpolation = cv2.INTER_LINEAR) # Before norm; for speed??
    if normalize:
        tile = cnorm.normalize(image = tile, target = None, 
                               verbose = False)
   
    cv2.imwrite(filename = filename, img = tile)


def update_map(tilemap, x, y, value):
    # Record a value at position (x,y) in the 2-array tilemap
    tilemap[y,x] = value

    return tilemap



##################################################################
##################################################################
###
###   ----  CREATING DATASETS FROM IMAGES IN A FOLDER ---- 
###
##################################################################
##################################################################


def flip(t):
    pass # make_data.py


def rotate(img, rotation_matrix):
    img = cv2.warpAffine(src = img, M = rotation_matrix, dsize= (img.shape[0:2]))
    return img


def data_rotate(t, iters, ext = 'jpg', mode = '3ch', writesize = 256):
    center = (writesize/2 - 1, writesize/2 - 1)
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle = 90, scale = 1.0)
    
    img_list = sorted(glob.glob(os.path.join(t, '*.'+ext)))
    for name in img_list:
        if mode == '3ch':
            img = cv2.imread(name)
        elif mode == '1ch':
            #img = cv2.imread(name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            img = cv2.imread(name)
            #img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)

        for k in range(iters):
            name = name.replace('.'+ext, 'r.'+ext)
            #print name
            img = rotate(img, rotation_matrix)
            cv2.imwrite(filename = name, img = img)
            
    print '\tDone rotating images in {}'.format(t)
   

def coloration(img, l_mean, l_std):
    target = np.array([[l_mean, l_std], [169.3, 9.01], [105.97, 6.67] ])
    return cnorm.normalize(img, target)


def data_coloration(t, mode, ext):
    # TODO replace with random  numbers generated from uniform distrib. 
    l_mean_range = (144.048, 130.22, 145.0, 135.5)
    l_std_range = (40.23, 35.00, 35.00, 37.5)

    img_list = sorted(glob.glob(os.path.join(t, '*.'+ext)))
    for idx, name in enumerate(img_list):
        if idx % 500 == 0:
            print "\tcolorizing {} of {}".format(idx, len(img_list))
        for LMN, LSTD in zip(l_mean_range, l_std_range):
            name_out = name.replace('.'+ext, 'c.'+ext)
            if mode == 'feat':
                img = cv2.imread(name)
                img = coloration(img, LMN, LSTD)
                cv2.imwrite(filename = name_out, img = img)
            elif mode == 'anno':
                img = cv2.imread(name)
                cv2.imwrite(filename = name_out, img = img)
    
    print '\tDone color augmenting images in {}'.format(t)
                


def writeList(src, anno):
    '''
    Take in a src dir, annotations dir, and a root folder.
    With digits this is unnecessary. 
    '''
    #listfile = os.path.join(t, 'list.txt')
    #with open(listfile, 'r') as f:
    pass



def split_img(t, ext, mode = "3ch", writesize = 256,tiles = 4):
    # Split into `tiles` number of sub-regions
    # It help if tiles is a square number.
    # Pull one image, get the dimensions:
    
    img_list = glob.glob(os.path.join(t, '*.'+ext))
    example = cv2.imread(img_list[0])
    h,w = example.shape[0:2]
   
    # TODO make this general, FFS!!!
    # added + 1 to make odd-numbered cuts
    # i.e it will have a true center. 
    grid = np.array([[0, 0, (w/2)+1, (h/2)+1],
                    [0, h/2, (w/2)+1, h],
                    [w/2, 0, w, (h/2)+1],
                    [w/2, h/2, w, h]])

    # remove the original image
    # name-out: tile1 --> tile1s, tile1ss, tile1sss, tile1ssss
    for name in img_list:
        if mode == "3ch":
            img = cv2.imread(name)
        elif mode == "1ch":
            #img = cv2.imread(name, 0)
            img = cv2.imread(name)
            #img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)

        os.remove(name)
        for i in range(tiles):
            r = grid[i, :]
            name = name.replace('.'+ext, 's.'+ext)

            if mode == "3ch":
                subimg = img[r[0]:r[2], r[1]:r[3], :]
                subimg = cv2.resize(subimg, dsize = (writesize, writesize),
                                    interpolation = cv2.INTER_NEAREST)
            elif mode == "1ch":
                subimg = img[r[0]:r[2], r[1]:r[3]]
                subimg = cv2.resize(subimg, dsize = (writesize, writesize), 
                                    interpolation = cv2.INTER_NEAREST)

            #subimg = cv2.resize(subimg, dsize = (writesize, writesize))
            cv2.imwrite(filename = name, img = subimg)

    print '\tDone partitioning images in {}'.format(t)                    
             

def random_crop(h, w, edge):
    minx = 0
    miny = 0

    maxx = w-edge
    maxy = h-edge

    x = np.random.randint(minx, maxx)
    y = np.random.randint(miny, maxy)

    x2 = x+edge
    y2 = y+edge

    return [x, x2, y, y2]


def sub_img(img_list, ext, mode = "3ch", edge = 512, writesize = 256, n = 8, coords = 0):
    # In contrast to split, do a random crop n times

    # img_list = sorted(glob.glob(os.path.join(path, '*.'+ext)))
    example = cv2.imread(img_list[0])
    h,w = example.shape[0:2]

    # Keep track of the randomly generated coordinates
    if coords == 0:
        # print 'Generating random coordinates'
        gencoord = True
        coords = [0]*len(img_list)
    else:
        # print 'Using imported coordinates'
        gencoord = False
    # print len(coords)

    for index, (name, c) in enumerate(zip(img_list, coords)):
        img = cv2.imread(name)
        # os.remove(name) # SUPER SKETCHYYYYY
        name = name.replace('.'+ext, '_{}.{}'.format(edge,ext))

        # print coord
        if gencoord:
            # Has to hold values up to max(h,w)
            coordsout = np.zeros(shape = (n, 4), dtype = np.uint32)
 
        for i in range(n):
            if gencoord:
                x, x2, y, y2 = random_crop(h,w, edge = edge)
                coordsout[i,:] = [x, x2, y, y2]
                # print 'Coordstout: ',coordsout[i,:], ' ', 
            else:
                x,x2,y,y2 = c[i,:]

            # print '{} : {} \t {} : {}'.format(x,x2,y,y2)
            name = name.replace('.'+ext, 's.'+ext)

            if mode == "3ch":
                subimg = img[x:x2, y:y2, :]
                subimg = cv2.resize(subimg, dsize = (writesize, writesize),
                                    interpolation = cv2.INTER_LINEAR)
            elif mode == "1ch":
                subimg = img[x:x2, y:y2, :]
                subimg = cv2.resize(subimg, dsize = (writesize, writesize), 
                                    interpolation = cv2.INTER_NEAREST)

            cv2.imwrite(filename = name, img = subimg)

            if gencoord:
                coords[index] = coordsout


    # print 'Done partitioning images in {}'.format(path)

    return coords


def partition_train_val(src, dst, trainpct = 0.85, valpct = 0.15):
    '''
    Training and validation partition
    Take care of a lot of preprocessing here

    1. Randomly partition data into training and validation according to the fractions
    2. Make new directories underneath 'dst'

    Assume src contains one dir for each class.
    Make sure the classes are evenly represented : 
    ---------------------------------------------
    '''
    pass


def delete_list(imglist):
    for img in imglist:
        os.remove(img)


def multiply_one_folder(src):
    '''
    I think this is a good idea. 1-26-17
    '''
    print "\nAffirm that \n {} is not the original dir.".format(src)
    choice = input("I have made copies (1) or not (anything else) \t");
    if choice == 1:
        print "Continuing"
    else:
        print "Make a copy of the data first. TODO make this less dumb."
        return 0 # break

    print "Rotating images in {}".format(src);
    data_rotate(src, 3, ext = 'jpg');
    print "Modulating color for data in {}.".format(src);
    data_coloration(src, 'feat', 'jpg');


def multiply_data(src, anno):
    '''
    Define a set of transformations, to be applied sequentially, to images.
    For each image, track it's annotation image and copy the relevant transformations.

    This should work for any sort fo experiment where 
    - annotation images are contained in one dir
    - similary named source images are contained in their own dir
    - we want them to be multiplied

    The goal is to not write a brand new script every time.
    '''
    print "\nAffirm that files in\n>{} \nand \n>{} \nare not originals.\n".format(src, anno) 
    choice = input("I have made copies. (1/no) ")

    if choice == 1:
        print "Continuing"
    else:
        print "non-1 response. exiting TODO: Make this nicer"
        return 0
    # That was important because the following functions write out to the original dirs.
    # I.E. they change the contents.. which is usually a no no. but this time is how it goes.
    
    srclist = sorted(glob.glob(os.path.join(src, '*.jpg')))
    annolist = sorted(glob.glob(os.path.join(anno, '*.png')))

    # Multi-scale
    for scale, numbersub in zip([756, 512, 256], [4, 12, 9]):
        coords = sub_img(srclist, ext = 'jpg', mode = "3ch", 
                         edge = scale, n = numbersub); 
        
        _ = sub_img(annolist, ext = 'png', mode = "1ch", 
                    edge = scale, coords = coords, n = numbersub);

    # Now it's oK to remove the originals
    delete_list(srclist)
    delete_list(annolist)

    data_coloration(src, 'feat', 'jpg'); 
    data_coloration(anno, 'anno', 'png');

    data_rotate(src, 3, ext = 'jpg', mode = "3ch"); 
    data_rotate(anno, 3, ext = 'png', mode = "1ch");
   


def find_bcg(wsi):
    pass # make_data.py


##################################################################
##################################################################
###
###       ~~~~~~~~~~~~~~~ TILES ~~~~~~~~~~~~~~~~~
###
##################################################################
##################################################################

def tile_wsi(wsi, tilesize, writesize, writeto, overlap = 0, prefix = 'tile'):

    lvl20x, dim20x = pull_svs_stats(wsi)
    resize_factor = int(wsi.level_downsamples[lvl20x]) # 1 if the slide is 20x
    
    # tilesize and overlap given w.r.t. 20X
    dims_top = wsi.level_dimensions[0]
    tile_top = int(dims_top[0] * tilesize / dim20x[0] / np.sqrt(resize_factor)) # (EQ 1)
    overlap_top = int(overlap * np.sqrt(resize_factor))

    print '[Output from : {}]'.format(PrintFrame())
    print "\ttilesize w.r.t. level 0 = {}".format(tile_top)
    print "\ttilesize w.r.t. 20x (level {}) = {}".format(lvl20x, tilesize)
    print "\tOverlap value w.r.t. level 0 = {}".format(overlap_top)
    print "\tOverlap value w.r.t. 20x (level {}) = {}".format(lvl20x, overlap)

    nrow = dims_top[1] / tile_top
    ncol = dims_top[0] / tile_top
    print '\tRounding off {} from rows and {} from cols'.format(
                                    dims_top[1] - nrow*tile_top,
                                    dims_top[0] - ncol*tile_top)
    
    # 3-17-17 test 0:nrow-1
    # Removing the first and last causes problems reassembling
    # The solution is to re-infer all these things at reassembly time
    # I guess that shouldn't be too hard but it's not something I want to explore
    # Besides there's really no problem with doing this on the whole thing so I'll keep it. 
    lst = [(k,j) for k in range(nrow) for j in range(ncol)]
    tilemap = np.zeros(shape = (nrow, ncol), dtype = np.uint32)
    ntiles = len(lst) # == nrow * ncol

    # TODO: add block read's and low-res whitespace mapping
    print ""
    print '[Output from : {}]'.format(PrintFrame()) 
    print "\tPopulating tilemap"
    print "\tCreated a {} row by {} col lattice over the image".format(nrow, ncol)
    print "\tPulling out squares side length = {}".format(tilesize + 2*overlap)
    print "\tWriting into {} squares".format(writesize)
    written = 0
    for index, coords in enumerate(lst):
        if index % 100 == 0:
            print "\t{:05d} / {:05d} ({} written so far)".format(index, ntiles, written)

        # Coordinates of tile's upper-left corner w.r.t. the predfined lattice
        [y, x] = coords
        name = '{}{:05d}.jpg'.format(prefix, index)
        tile = wsi.read_region(location = (x*tile_top - overlap_top, y*tile_top - overlap_top), 
                               level = 0, 
                               size =(tile_top + 2*overlap_top, tile_top + 2*overlap_top)) 
        tile = np.array(tile)

        if check_white(tile):
            filename = os.path.join(writeto, name)
            write_tile(tile, filename, writesize, normalize = False)
            tilemap = update_map(tilemap, x, y, index)
            written += 1

    return tilemap



# TODO fix this logic. It's not good. 
def create_dirs_inference(filename, writeto, sub_dirs, remove = False):
    # Really, this asks to remove the 'Tile' root directory
    # We're always going to clean out the requested sub-dirs.
    tail = os.path.basename(filename)
    slide_name, ex = os.path.splitext(tail)
    exp_home = os.path.join(writeto, slide_name)
    use_existing = True
    fresh_dir = False

    created_dirs = [os.path.join(exp_home, d) for d in sub_dirs]

    # Take care of root first:
    if remove and os.path.exists(exp_home):
        # Clean all of them, even Tile
        print '\tCleaning up {}'.format(exp_home)
        _ = [shutil.rmtree(d) for d in created_dirs if os.path.exists(d)]
        # shutil.rmtree(exp_home) # Can't just haphazardly delete everything that's dum
        use_existing = False
        fresh_dir = True

    if not os.path.exists(exp_home):
        os.makedirs(exp_home)
        fresh_dir = True
        use_existing = False

    if fresh_dir:
        _ = [os.mkdir(d) for d in created_dirs]
    else:
        # Clear the ones that aren't tile:
        print '\tPartially cleaning in {}'.format(exp_home)
        _ = [shutil.rmtree(d) for d in created_dirs[1:] if os.path.exists(d)]
        _ = [os.mkdir(d) for d in created_dirs[1:]]
    return exp_home, created_dirs, use_existing


# New: adding overlap option
'''
    Method for overlapping:
        - Create lattice without considering overlap
        - Add overlap to tilesize in both dims
        - Writesize remains 256; that's what the network wants. 
        (change this by not being dum)
'''
def make_inference(filename, writeto, create, tilesize = 512, 
                   writesize = 256, overlap = 0, remove_first = False):
    
    exp_home, created_dirs, use_existing = create_dirs_inference(filename, 
                                                                 writeto, 
                                                                 sub_dirs = create, 
                                                                 remove = remove_first) 
    tiledir = created_dirs[0]

    if use_existing:
        # TODO add in here more feedback for this tree of action.
        # TODO add here checks to see if the written tiles match the requested tiles.

        print "\tUsing existing tiles located {}".format(tiledir)
        for d in created_dirs[1:]:
            print "\tCreated: {}".format(d)

        return tiledir, exp_home, created_dirs[1:]

    for d in created_dirs:
        print "\tCreated: {}".format(d)

    wsi = OpenSlide(filename)
    print "\tWorking with slide {}".format(filename)
    
    # Echo the settings:
    print ''
    print '[Output from : {}]'.format(PrintFrame()) 
    print "\tSettings -------------"
    print "\tTilesize: {}".format(tilesize)
    print "\tWrite size: {}".format(writesize)
    print "\tOverlap: {}".format(overlap)
    print "\t ------------- end/Settings\n"
    tilemap = tile_wsi(wsi, tilesize, writesize, tiledir, overlap, prefix = 'tile')

    # Write out map file as npy
    map_file = os.path.join(exp_home, 'data_tilemap_{}.npy'.format(tilesize))
    np.save(file = map_file, arr = tilemap)

    wsi.close()

    #returns created_dirs after the first, which should always be 'tile'
    return exp_home, created_dirs


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
    

def get_all_regions(m, threshold = 80):
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

def empty_block(place_size):
    return np.zeros(shape = (place_size, place_size, 3), dtype = np.uint8)


def load_block(pth, place_size, overlap, interp = cv2.INTER_LINEAR):
    # process overlapping borders...
    # 1. Upsample block to be place_size + 2*overlap
    # 2. Remove the overlapping parts
   
    block = cv2.imread(pth)
    ds_size = block.shape[0]
    content = ds_size - 2*overlap
    block = block[overlap : -overlap, overlap : -overlap, :]

    block = cv2.resize(block, dsize = (place_size, place_size), 
                       interpolation = interp)

    return block


def overlay_colors(img, block):
    img = np.add(img*0.6, block*0.4)
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
            pth = os.path.join(source_dir, 'tile{:05d}.png'.format(r))

            block = load_block(pth, place_size, overlap)

            if block is None:
                print '[Output from : {}]'.format(PrintFrame())
                print "\timg {} ought to exist but I guess doesn't.".format(pth)
                block = empty_block(place_size)

            if do_overlay:
            # TODO fix the hard-coded "tile" prefix and extension suffix: 
                ov_pth = os.path.join(overlay_dir, 'tile{:05d}.jpg'.format(r))
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
    

def downsize_add_padding(img, target, pad = (0,0), interp = cv2.INTER_AREA):
    # First:
    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tOriginal shape {}'.format(img.shape)
    print '\tTarget dimensions (final): {}'.format(target)
    print '\tPad at target scale: {}'.format(pad)

    r,c,_ = img.shape
    tr, tc = target 
    pr, pc = pad

    # resize to the right scale
    print '\tGetting padding at reassembled scale'
    dc = float(c) / tc
    dr = float(r) / tr
    pr = int(pr * dr)
    pc = int(pc * dc)
    print '\tScaled pad: row: {} col: {}'.format(dr, dc)

    padc = np.zeros(shape = (r, pc, 3))
    print '\tImg: {} padc: {}'.format(img.shape, padc.shape)
    img = np.hstack((img, padc)) 

    padr = np.zeros(shape = (pr, c+pc, 3))
    print '\tImg: {} padr: {}'.format(img.shape, padr.shape)
    img = np.vstack((img, padr))

    img = cv2.resize(img, (tc, tr), interpolation = interp)
    print '\tFinal size: {}'.format(img.shape)
    return img


def partition_rows(m, h):
    rows = []
    for ix in range(h):
        rows.append(m[ix, :])
    return rows


def build_region(region, m, source_dir, place_size, overlap, 
                 overlay_dir, max_w = 10000, exactly = None, pad = (0,0)):
    x,y,c,r = region
    print ''
    print '[Output from : {}]'.format(PrintFrame()) 
    print "\tRegion source: {}".format(source_dir)
    print "\tx: {} y: {} c: {} r: {}".format(x,y,c,r)

    # Check if the output will be large
    if c*r*(place_size**2) > (2**31)/3:
        # edit place_size so that the output will fit:
        print '[Output from : {}]'.format(PrintFrame())
        print "\tFound region > 2**31, resizing to ",
        place_size = int(np.sqrt(((2**31)/3) / (c*r)))
        print "\t{}".format(place_size)

    print '\tm is : {}'.format(m.shape)
    rows = partition_rows(m[y:y+r, x:x+c], r)
    print '\tFound {} rows'.format(len(rows)) 

    built_img = [] # Not really an image; a list of row images
    for ix, row in enumerate(rows):
        # print '\tRow {}/{}'.format(ix, len(rows))
        row_ = build_row(row, source_dir, place_size, overlap, overlay_dir)
        built_img.append(row_)

    img = assemble_rows(built_img)

    if exactly is None:
        if img.shape[1] > max_w:
            print '[Output from : {}]'.format(PrintFrame())
            print "\tFound w > {}; resizing".format(max_w) 
            img = downsize_keep_ratio(img, max_w, interp = cv2.INTER_AREA)
    elif any(img.shape[:2] != exactly):
        print ''
        print '[Output from : {}]'.format(PrintFrame())
        print "\tFound {} != {}; resizing".format(img.shape[:2], exactly) 
        # c_, r_ = exactly
        img = downsize_add_padding(img, exactly, pad)

    return img


def calc_tile_cutoff(filename, tilesize):
    # Need to find how many tiles make a reasonable area 
    # Given the level - 0 dimensions
    f = OpenSlide(filename)
    lvl0 = f.level_dimensions[0]
    n_tiles = (lvl0[0]/tilesize) * (lvl0[1]/tilesize)

    return n_tiles / 20



# TODO this function isn't very good. the place to generalize isn't really obvious to me. 
def assemble(exp_home, expdirs, writesize, overlap, overlay, area_cutoff, tilesize):
    
    # Force sources to be a list:
    if isinstance(expdirs, basestring):
        expdirs = [expdirs]

    # Pull in the image map:
    map_file = os.path.join(exp_home, 'data_tilemap_{}.npy'.format(tilesize))
    m = np.load(map_file)
    [N,M] = m.shape # Forego printing

    # Pull out disconnected regions that pass a size cutoff:
    regions = get_all_regions(m, threshold = area_cutoff)

    overlay_dir = ''
    if overlay:
        # Tiles overlaid onto tiles should work fine.
        overlay_dir = os.path.join(exp_home, expdirs[0])

    for index, reg in enumerate(regions):
        # TODO this loop still sucks
        #print PrintFrame(),"Processing region {}".format(index)
        for src in expdirs[1:]:
            src_base = os.path.basename(src) 
            reg_name = '{:03d}_{}.jpg'.format(index, src_base) 
            
            # if 'prob' in src_base:
            #     overlay_dir_use = ''  
            # else:
            #     overlay_dir_use = overlay_dir

            img = build_region(reg, m, src, writesize, overlap, overlay_dir) 
            
            reg_name = os.path.join(exp_home, reg_name)
            print "\tSaving region to {}".format(reg_name)
            cv2.imwrite( reg_name, img )


