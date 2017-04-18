'''
Preprocessing for the whole slide image

includes separating tissue from background
and processable from non-processable tissue

use a white threshold to determine tissue / background
use an AlexNet to determine processable / non-processable tissue

essentially I want a function that takes in WSI
and spits out tilemap
    Make the function agnostic to the tilesize
    i.e. return a very rough map, that will be later reshaped
    to fit over tilemap
    The row:col ratio should be similar

    favor over-calling tissue area.
    favor a liberal definition of processable
        really exclude:
            fatty tissue
            tears
            edges
            folds
            marker ***
        Use a different sort of algorithm for each and then at the end
        just take the union.


changelog
---------
2017-04-18:
    Initial

'''



'''
Functions

tissue_map = find_tissue(wsi, **kwargs)
tumor_map = find_tumor(wsi, **kwargs)

'''

def find_tissue(wsi, **kwargs):
    pass

def find_tumor(wsi, **kwargs):
    pass

def processable(img, **kwargs):
    pass

def is_white_space(img, **kwargs):
    pass

def main():
    pass


if __name__ == '__main__':
    main()




