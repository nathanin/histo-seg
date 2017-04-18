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
import time
'''
```````````````` DEBUGGING FUNCTOIN ``````````````````
'''


def PrintFrame():
    callerframerecord = inspect.stack()[1]  #0 represents this line
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    thisfile = info.filename
    thisfun = info.function
    thisline = info.lineno

    return '{} in {} @ {}'.format(thisfile, thisfun, thisline)


'''
```````````````` DEBUGGING FUNCTOIN ``````````````````
'''

##################################################################
##################################################################
###
###   ----  CREATING DATASETS FROM IMAGES IN A FOLDER ----
###
##################################################################
##################################################################


def flip(t):
    pass  # make_data.py


def rotate(img, rotation_matrix):
    img = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(img.shape[0:2]))
    return img


def data_rotate(t, iters, ext='jpg', mode='3ch', writesize=256):
    center = (writesize / 2 - 1, writesize / 2 - 1)
    rotation_matrix = cv2.getRotationMatrix2D(
        center=center, angle=90, scale=1.0)

    img_list = sorted(glob.glob(os.path.join(t, '*.' + ext)))
    for name in img_list:
        if mode == '3ch':
            img = cv2.imread(name)
        elif mode == '1ch':
            #img = cv2.imread(name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            img = cv2.imread(name)
            #img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)

        for k in range(iters):
            name = name.replace('.' + ext, 'r.' + ext)
            #print name
            img = rotate(img, rotation_matrix)
            cv2.imwrite(filename=name, img=img)

    print '\tDone rotating images in {}'.format(t)


def coloration(img, l_mean, l_std):
    target = np.array([[l_mean, l_std], [169.3, 9.01], [105.97, 6.67]])
    return cnorm.normalize(img, target)


def data_coloration(t, mode, ext):
    # TODO replace with random  numbers generated from uniform distrib.
    l_mean_range = (144.048, 130.22, 135.5, 140.0)
    l_std_range = (40.23, 35.00, 35.00, 37.5)

    img_list = sorted(glob.glob(os.path.join(t, '*.' + ext)))
    for idx, name in enumerate(img_list):
        if idx % 500 == 0:
            print '\tcolorizing {} of {}'.format(idx, len(img_list))
            for LMN, LSTD in zip(l_mean_range, l_std_range):
                name_out = name.replace('.' + ext, 'c.' + ext)
                if mode == 'feat':
                    img = cv2.imread(name)
                    img = coloration(img, LMN, LSTD)
                    cv2.imwrite(filename=name_out, img=img)
                elif mode == 'anno':
                    img = cv2.imread(name)
                    cv2.imwrite(filename=name_out, img=img)

    print '\tDone color augmenting images in {}'.format(t)


def writeList(src, anno):
    '''
    Take in a src dir, annotations dir, and a root folder.
    With digits this is unnecessary.
    '''
    #listfile = os.path.join(t, 'list.txt')
    #with open(listfile, 'r') as f:
    pass


def random_crop(h, w, edge):
    minx = 0
    miny = 0
    maxx = w - edge
    maxy = h - edge
    x = np.random.randint(minx, maxx)
    y = np.random.randint(miny, maxy)
    x2 = x + edge
    y2 = y + edge

    return [x, x2, y, y2]


def sub_img(img_list, ext, mode='3ch', edge=512, writesize=256, n=8, coords=0):
    # In contrast to split, do a random crop n times

    # img_list = sorted(glob.glob(os.path.join(path, '*.'+ext)))
    example = cv2.imread(img_list[0])
    h, w = example.shape[0:2]

    # Keep track of the randomly generated coordinates
    if coords == 0:
        gencoord = True
        coords = [0] * len(img_list)
    else:
        gencoord = False

    for index, (name, c) in enumerate(zip(img_list, coords)):
        img = cv2.imread(name)
        name = name.replace('.' + ext, '_{}.{}'.format(edge, ext))

        # print coord
        if gencoord:
            coordsout = np.zeros(shape=(n, 4), dtype=np.uint32)

        for i in range(n):
            if gencoord:
                x, x2, y, y2 = random_crop(h, w, edge=edge)
                coordsout[i, :] = [x, x2, y, y2]
            else:
                x, x2, y, y2 = c[i, :]

            name = name.replace('.' + ext, 's.' + ext)

            if mode == '3ch':
                subimg = img[x:x2, y:y2, :]
                subimg = cv2.resize(
                    subimg,
                    dsize=(writesize, writesize),
                    interpolation=cv2.INTER_LINEAR)
            elif mode == '1ch':
                subimg = img[x:x2, y:y2, :]
                subimg = cv2.resize(
                    subimg,
                    dsize=(writesize, writesize),
                    interpolation=cv2.INTER_NEAREST)

            # this is always going to write
            # linter places it in there with the others
            cv2.imwrite(filename=name, img=subimg)

            if gencoord:
                coords[index] = coordsout

    return coords


def delete_list(imglist):
    print 'Removing {} files'.format(len(imglist))
    for img in imglist:
        os.remove(img)


def multiply_one_folder(src):
    '''
    I think this is a good idea. 1-26-17
    '''
    print '\nAffirm that \n {} is not the original dir.'.format(src)
    choice = input('I have made copies (1) or not (anything else) \t')
    if choice == 1:
        print 'Continuing'
    else:
        print 'Make a copy of the data first. TODO make this less dumb.'
        return 0  # break

    print 'Rotating images in {}'.format(src)
    data_rotate(src, 3, ext='jpg')
    print 'Modulating color for data in {}.'.format(src)
    data_coloration(src, 'feat', 'jpg')


def multiply_data(src, anno, scales = [512], multiplicity = [9]):
    '''
    Define a set of transformations, to be applied sequentially, to images.
    For each image, track it's annotation image and copy the relevant transformations.

    This should work for any sort fo experiment where
    - annotation images are contained in one dir
    - similary named source images are contained in their own dir
    - we want them to be multiplied

    '''

    print '\nAffirm that files in\n>{} \nand \n>{} \nare not originals.\n'.format(
        src, anno)
    choice = input('I have made copies. (1/no) ')

    if choice == 1:
        print 'Continuing'
    else:
        print 'non-1 response. exiting TODO: Make this nicer'
        return 0

    if len(scales) != len(multiplicity):
        print 'Warning: scales and multiplicity must match lengths'
        return 0

    srclist = sorted(glob.glob(os.path.join(src, '*.jpg')))
    annolist = sorted(glob.glob(os.path.join(anno, '*.png')))

    # Multi-scale
    for scale, numbersub in zip(scales, multiplicity):
        print 'Extracting {} subregions of size {}'.format(numbersub, scale)
        coords = sub_img(
            srclist, ext='jpg', mode='3ch', edge=scale, n=numbersub)
        print 'Repeating for png'
        _ = sub_img(
            annolist,
            ext='png',
            mode='1ch',
            edge=scale,
            coords=coords,
            n=numbersub)


    # Now it's OK to remove the originals
    delete_list(srclist)
    delete_list(annolist)

    data_coloration(src, 'feat', 'jpg')
    data_coloration(anno, 'anno', 'png')

    data_rotate(src, 3, ext='jpg', mode='3ch')
    data_rotate(anno, 3, ext='png', mode='1ch')


def find_bcg(wsi):
    pass  # make_data.py


##################################################################
##################################################################
###
###       ~~~~~~ Inference from SVS file ~~~~~~~~~
###
##################################################################
##################################################################


def pull_svs_stats(svs):
    # Specficially for svs files
    app_mag = svs.properties['aperio.AppMag']
    level_dims = svs.level_dimensions

    # Omit print statement

    # Find 20X level:
    if app_mag == '20':  # scanned @ 20X
        return 0, level_dims[0]
    if app_mag == '40':  # scanned @ 40X
        return 1, level_dims[1]


def check_white(tile, cutoff=215, pct=0.75):
    # True if the tile is mostly non-white
    # Lower pct ~ more strict
    gray = cv2.cvtColor(tile, cv2.COLOR_RGBA2GRAY)
    white = (gray > cutoff).sum() / float(gray.size)

    return white < pct


def write_tile(tile, filename, writesize, normalize):
    # Needed for inference by SegNet
    tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)  # ???
    tile = tile[:, :, (2, 1, 0)]  # ???

    tile = cv2.resize(
        tile, dsize=(writesize, writesize),
        interpolation=cv2.INTER_LINEAR)  # Before norm; for speed??
    if normalize:
        tile = cnorm.normalize(image=tile, target=None, verbose=False)

    cv2.imwrite(filename=filename, img=tile)


# Record a value at position (x,y) in the 2-array tilemap
def update_map(tilemap, r, c, value):
    tilemap[r, c] = value
    return tilemap


def nrow_ncol(wsi, tilesize, overlap):
    lvl20x, dim20x = pull_svs_stats(wsi)
    resize_factor = int(wsi.level_downsamples[lvl20x])

    dims_top = wsi.level_dimensions[0]
    tile_top = int(
        dims_top[0] * tilesize / dim20x[0] / np.sqrt(resize_factor))
    overlap_top = int(overlap * np.sqrt(resize_factor))

    nrow = dims_top[1] / tile_top
    ncol = dims_top[0] / tile_top

    return tile_top, overlap_top, nrow, ncol


# New Apr 12, 2017
def locate_tumor(wsi, tilesize, overlap):
    # Return a tilemap
    # Get some info about the slide
    tile_top, nrow, ncol = nrow_ncol(wsi, tilesize, overlap)
    tilemap = np.zeros(shape=(nrow, ncol), dtype=np.bool)
    lst = [(k, j) for k in range(1, nrow - 1) for j in range(1, ncol - 1)]

    # Try a quick threshold
    lowres = wsi.level_dimensions[-1]



'''
    tile_wsi write out tiles and save a map
    # Functionized. 4-12-17. NI
    # Debugged 4-17-17
    # Added timing 4-17-17

    1. If tilemap is given, use it to determine valid locations
    2. For each location, decide if this location is valid based
        on tumor_located
    3. If tumor_located == True,
        check for white space
    4. If not white space,
        write out tile and update tilemap

'''

def tile_wsi(wsi, tilesize, writesize, writeto, overlap=0, prefix='tile',
             tilemap=None):
    start_time = time.time()

    # Do some conversions based on the presumed tile size,
    # translate it to the top level where we always take data from.
    tile_top, overlap_top, nrow, ncol = nrow_ncol(wsi, tilesize, overlap)

    # There's probably some clever way to write this
    if tilemap:
        tumor_located = tilemap
        # Reset tilemap
        tilemap = np.zeros(shape=(nrow, ncol), dtype=np.uint32)
    else:
        tilemap = np.zeros(shape=(nrow, ncol), dtype=np.uint32)
        tumor_located = np.ones(shape=(nrow, ncol), dtype=np.bool)
        #tumor_located = tilemap + 1
        #tumor_located.dtype = np.bool

    # The (0,0) coordinate now is eqivalent to (tilesize-overlap, tilesize-overlap)
    lst = [(k, j) for k in range(1, nrow - 1) for j in range(1, ncol - 1)]

    ntiles = len(lst)  # == nrow * ncol

    written = 0
    for index, coords in enumerate(lst):
        # Incrementally print some feedback
        if index % 100 == 0:
            print '\t{:05d} / {:05d} ({} written so far)'.format(
                index, ntiles, written)

        # Coordinates of tile's upper-left corner w.r.t. the predfined lattice
        [r, c] = coords
        # First check
        # For new tilemaps, this is always True
        # Meant to minimze calls to read_region
        if tumor_located[r,c]:
            name = '{}{:05d}.jpg'.format(prefix, index)
            tile = wsi.read_region(
                location=(c * tile_top - overlap_top, r * tile_top - overlap_top),
                level=0,
                size=(tile_top + 2 * overlap_top, tile_top + 2 * overlap_top))
            tile = np.array(tile)

            # Second check for white space
            if check_white(tile):
                filename = os.path.join(writeto, name)
                write_tile(tile, filename, writesize, normalize=True)
                tilemap = update_map(tilemap, r, c, index)

                # Increment our dumb counter
                written += 1

    # Print out timing info
    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME data.tile_wsi tilesize: {} time: {}'.format(
        tilesize, elapsed)

    return tilemap



def create_dirs_inference(filename, writeto, sub_dirs, remove=False):
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
        # Leave the following here as a warning
        # Can't just haphazardly delete everything that's dum
        # shutil.rmtree(exp_home)
        use_existing = False
        fresh_dir = True

    if not os.path.exists(exp_home):
        os.makedirs(exp_home)
        use_existing = False
        fresh_dir = True
    else:
        # Flag anything that's missing
        for d in created_dirs:
            if not os.path.exists(d):
                use_existing = False
                fresh_dir = True

    if fresh_dir:
        print '\tCreating the sub-directory tree'
        [os.mkdir(d) for d in created_dirs if not os.path.exists(d)]
    else:
        # Clear the ones that aren't tile:
        print '\tPartially cleaning in {}'.format(exp_home)
        _ = [shutil.rmtree(d) for d in created_dirs[1:] if os.path.exists(d)]
        _ = [os.mkdir(d) for d in created_dirs[1:]]

    return exp_home, created_dirs, use_existing


# New: adding overlap option
def make_inference(filename,
                   writeto,
                   create,
                   tilesize=512,
                   writesize=256,
                   overlap=0,
                   remove_first=False):

    start_time = time.time()
    exp_home, created_dirs, use_existing = create_dirs_inference(
        filename, writeto, sub_dirs=create, remove=remove_first)
    tiledir = created_dirs[0]

    if use_existing:
        # TODO add in here more feedback for this tree of action.
        # TODO add here checks to see if the written tiles match the requested tiles.
        print '\tUsing existing tiles located {}'.format(tiledir)
        for d in created_dirs[1:]:
            print '\tCreated: {}'.format(d)

        return tiledir, created_dirs

    for d in created_dirs:
        print '\tCreated: {}'.format(d)

    # Get the slide pointer
    wsi = OpenSlide(filename)
    print '\tWorking with slide {}'.format(filename)

    # TODO (nathan) low-level tumor location
    #tilemap = locate_tumor(wsi)
    tilemap = tile_wsi(
        wsi, tilesize, writesize, tiledir, overlap, prefix='tile')

    # Write out map file as npy
    map_file = os.path.join(exp_home, 'data_tilemap_{}.npy'.format(tilesize))
    np.save(file=map_file, arr=tilemap)

    wsi.close()

    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME data.make_inference time: {}'.format(elapsed)

    #returns created_dirs after the first, which should always be 'tile'
    return exp_home, created_dirs


def label_regions(m):
    h, w = m.shape
    bw = m > 0
    bw.dtype = np.uint8  # can't keep bool

    contours, _ = cv2.findContours(
        image=bw, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    ll = np.zeros(shape=(h, w), dtype=np.uint8)
    for idx, ct in enumerate(contours):
        cv2.drawContours(
            ll, [ct], 0, idx + 1, thickness=-1)  # thickness -1 means to fill

    return ll, contours


def get_all_regions(m, threshold=80):
    '''
    Spit out regions as bounding rects of connected components
    threshold is the number of tiles required (should be careful of large tilesizes)
    '''
    regions = []
    ll, contours = label_regions(m)

    for idx, ct in enumerate(contours):
        if (ll == idx + 1).sum() >= threshold:
            regions.append(cv2.boundingRect(ct))

    return regions


def empty_block(place_size):
    return np.zeros(shape=(place_size, place_size, 3), dtype=np.uint8)


def load_block(pth, place_size, overlap, interp=cv2.INTER_LINEAR):
    # process overlapping borders...
    # 1. Remove the overlapping parts
    # 2. Upsample block to be place_size + 2*overlap

    if not os.path.exists(pth):
        return None

    block = cv2.imread(pth)  # 256 x 256
    ds_size = block.shape[0]
    content = ds_size - 2 * overlap  # 256 - ds_overlap
    if overlap > 0:
        block = block[overlap:-overlap, overlap:-overlap, :]

    ## Upsample block to be exactly place_size, after removing the overlap.
    block = cv2.resize(
        block, dsize=(place_size, place_size), interpolation=interp)
    return block


def overlay_colors(img, block):
    img = np.add(img * 0.6, block * 0.4)
    img = cv2.convertScaleAbs(img)

    return img


def build_row(row, source_dir, place_size, overlap, overlay_dir=''):
    do_overlay = os.path.exists(overlay_dir)

    def load_or_empty(r):
        if r == 0:
            block = empty_block(place_size)
        else:
            pth = os.path.join(source_dir, 'tile{:05d}.png'.format(r))
            block = load_block(pth, place_size, overlap)

            if block is None:
                print '[Output from : {}]'.format(PrintFrame())
                print '\timg {} ought to exist but I guess doesnt.'.format(pth)
                block = empty_block(place_size)

            if do_overlay:
                # TODO fix the hard-coded 'tile' prefix and extension suffix:
                ov_pth = os.path.join(overlay_dir, 'tile{:05d}.jpg'.format(r))
                ov_img = load_block(ov_pth, place_size, overlap)
                ov_img = cv2.resize(ov_img, dsize=(place_size, place_size))
                block = overlay_colors(ov_img, block)

        return block

    #TODO See if we can do it without initializing with empty col on the left :
    row_out = [load_or_empty(r) for r in row]
    row_out = np.hstack(row_out)
    return row_out


def assemble_rows(rows):
    img = rows[0]
    for r in rows[1:]:
        img = np.append(img, r, axis=0)

    return img


def downsize_keep_ratio(img, dim=1, target_dim=1024, interp=cv2.INTER_NEAREST):
    # dx, dy seems to not work in python implementation .. ?
    # Since i need this to fit exact numbers, usually i think it's
    # a little safer to use the literal values for (r, c)
    # to avoid rounding errors
    factor = img.shape[dim] / float(target_dim)
    if dim == 1:
        target_h = int(img.shape[0] / factor)
        img_new = cv2.resize(
            src=img, dsize=(target_dim, target_h), interpolation=interp)
    elif dim == 0:
        target_w = int(img.shape[1] / factor)
        img_new = cv2.resize(
            src=img, dsize=(target_w, target_dim), interpolation=interp)
        return img_new


def partition_rows(m, h):
    rows = []
    for ix in range(h):
        rows.append(m[ix, :])

    return rows


def add_padding(img, target):
    print ''
    print '[Output from : {}]'.format(PrintFrame())
    print '\tImage: {}'.format(img.shape)
    print '\tTarget: {}'.format(target)
    y, x = target
    h, w = img.shape[:2]

    diffr, diffc = [y - h, x - w]
    padc = np.zeros(shape=(h, diffc, 3))
    padr = np.zeros(shape=(diffr, x, 3))

    print '\tColumn pad: {}'.format(padc.shape)
    print '\tRow pad: {}'.format(padr.shape)

    img = np.hstack((img, padc))
    img = np.vstack((img, padr))

    print '\tFinal shape {}'.format(img.shape)

    return img


def build_region(region,
                 m,
                 source_dir,
                 place_size,
                 overlap,
                 overlay_dir,
                 max_w=10000,
                 exactly=None,
                 pad=(0, 0)):

    start_time = time.time()
    x, y, c, r = region


    #print ''
    #print '[Output from : {}]'.format(PrintFrame())
    #print '\tRegion source: {}'.format(source_dir)
    #print '\tx: {} y: {} c: {} r: {}'.format(x, y, c, r)
    #print '\tPlacing in tiles: {}'.format(place_size)
    #print '\tUsing level overlap = {}'.format(overlap)

    # Check if the output will be large
    if c * r * (place_size**2) > (2**31) / 3:
        # edit place_size so that the output will fit:
        print '[Output from : {}]'.format(PrintFrame())
        print '\tFound region > 2**31, resizing to ',
        place_size = int(np.sqrt(((2**31) / 3) / (c * r)))
        print '\t{}'.format(place_size)

    print '\tm is : {}'.format(m.shape)
    rows = partition_rows(m[y:y + r, x:x + c], r)
    print '\tFound {} rows'.format(len(rows))

    built_img = []  # Not really an image; a list of row images
    for ix, row in enumerate(rows):
        row_ = build_row(row, source_dir, place_size, overlap, overlay_dir)
        built_img.append(row_)

    img = assemble_rows(built_img)

    print '\tImage shape: {}'.format(img.shape)
    print '\tExact shape: {}'.format(exactly)
    if exactly is None:
        if img.shape[1] > max_w:
            img = downsize_keep_ratio(
                img, dim=1, target_dim=max_w, interp=cv2.INTER_AREA)
    elif not img.shape[:2] == exactly:
        #img = add_padding(img, exactly)
        # ??????
        img = cv2.resize(img, dsize=exactly[::-1])

    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME data.build_region time: {}'.format(elapsed)

    return img


def calc_tile_cutoff(filename, tilesize):
    # Need to find how many tiles make a reasonable area
    # Given the level - 0 dimensions
    f = OpenSlide(filename)
    lvl0 = f.level_dimensions[0]
    n_tiles = (lvl0[0] / tilesize) * (lvl0[1] / tilesize)

    return n_tiles / 20


def assemble(exp_home, expdirs, writesize, overlap, overlay, area_cutoff,
             tilesize):

    # Force sources to be a list:
    if isinstance(expdirs, basestring):
        expdirs = [expdirs]

    # Pull in the image map:
    map_file = os.path.join(exp_home, 'data_tilemap_{}.npy'.format(tilesize))
    m = np.load(map_file)
    [N, M] = m.shape  # Forego printing

    # Pull out disconnected regions that pass a size cutoff:
    regions = get_all_regions(m, threshold=area_cutoff)

    overlay_dir = ''
    if overlay:
        # Tiles overlaid onto tiles should work fine.
        overlay_dir = os.path.join(exp_home, expdirs[0])

    for index, reg in enumerate(regions):
        # TODO this loop still sucks
        for src in expdirs[1:]:
            src_base = os.path.basename(src)
            reg_name = '{:03d}_{}.jpg'.format(index, src_base)

            # if 'prob' in src_base:
            #     overlay_dir_use = ''
            # else:
            #     overlay_dir_use = overlay_dir

            img = build_region(reg, m, src, writesize, overlap, overlay_dir)

            reg_name = os.path.join(exp_home, reg_name)
            print '\tSaving region to {}'.format(reg_name)
            cv2.imwrite(reg_name, img)


if __name__ == '__main__':
    import seg_pipeline  # This actually iports this function. idk.

    writeto = '/home/nathan/histo-seg/pca/seg_0.8.1024'
    sub_dirs = ['tiles', 'result', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4']

    # For multiscale, these aren't needed.
    weights = 'dummy'
    model_template = 'dummy'
    remove = True
    overlap = 64
    tilesize = 512
    writesize = 256

    filename = '/home/nathan/data/pca_wsi/1305400.svs'

    print 'Entering tile procedure...'
    seg_pipeline.run_multiscale(
        filename=filename,
        writeto=writeto,
        sub_dirs=sub_dirs,
        tilesize=tilesize,
        writesize=writesize,
        weights=weights,
        model_template=model_template,
        remove_first=remove,
        overlap=overlap,
        nclass=5,
        whiteidx=3,
        tileonly=True)


