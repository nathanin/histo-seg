#!/usr/bin/python

###
# Clean up all folders #
###

import os
import shutil


def lstdirs(lookin='.'):
    dlist = os.listdir(lookin)
    return [d for d in os.listdir(lookin) if os.path.isdir(d)]


def purge(dlist):
    for d in dlist:
        print 'Removing {}'.format(d)
        shutil.rmtree(d)


if __name__ == '__main__':
    response = raw_input('Are you sure? [y,n] ')

    if response == 'y':
        print "Cleaning up...."
        dl = sorted(lstdirs())

        print "Found {} folders to delete:".format(len(dl))
        for d in dl:
            print d

        response = raw_input('Are you sure? [y,n] ')

        if response == 'y':
            print 'OK.'
            purge(dl)
