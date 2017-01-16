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

def pull_svs_stats(svs):
    pass # tile_wsi.py

def check_white(tile, cutoff = 225, pct = 0.5):
    pass # tile_wsi.py

def write_tile(tile, filename, writesize):
    pass # tile_wsi.py

def update_map(tilemap, x, y, value):
    pass # tile_wsi.py

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
    pass # tile_wsi.py
