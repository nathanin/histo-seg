#!/usr/bin/python

import os
import glob
import shutil

def cleanup(d):
    pass

def make_lists(dirlist, image_name, dest, suffix):
    source_list = [os.path.join(d, image_name) for d in dirlist]
    source_list = [i for i in source_list if os.path.exists(i)]

    dirbases = [os.path.basename(d) for d in dirlist]
    dest_list = [os.path.join(dest, '{}_{}'.format(d, suffix)) for d in dirlist]

    return source_list, dest_list


def print_ab(a, b):
    print '{} \t--->\t {}'.format(a, b)


def copyprint(a, b):
    print_ab(a, b)
    shutil.copyfile(a, b)


def main(source='.', dest='.', image_name='multiscale_colored.jpg'):
    pwd = os.getcwd()
    os.chdir(source)
    dirlist = [d for d in os.listdir('.') if os.path.isdir(d)]
    ndir = len(dirlist)

    print 'Found {} directories under {}'.format(ndir, source)
    print 'Looking for `{}` under source directory tree'.format(image_name)

    source_list, dest_list = make_lists(dirlist, image_name, dest, image_name)

    print 'Copying {} files'.format(len(source_list))
    [copyprint(s, d) for s, d in zip(source_list, dest_list)]
    os.chdir(pwd)


if __name__ == '__main__':
    source = './triplet'
    dest_root = '/home/nathan/Dropbox/projects/semantic_pca/wsi_results'
    dest = os.path.join(dest_root, 'triplet')

    if not os.path.exists(dest): os.mkdir(dest)

    # Hard coded names of results images
    main(source, dest, 'report.pdf')
    #main(source, dest, 'report.txt')
    # main(source, dest, 'class_images.pdf')

