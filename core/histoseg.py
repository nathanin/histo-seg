'''
aka inference.py

Run an FCN or SegNet with trained and normalized weights.

'''
import os
import sys
import glob
import cv2
import inspect
import numpy as np
import generate_color

try:
    CAFFE_ROOT = '/home/nathan/caffe-segnet-crf'
    sys.path.insert(0, CAFFE_ROOT + "/python")
    import caffe
except:
    print "HISTOSEG import error: caffe"

import time
# Define inspection code that spits out the line it's called from (as str)


def PrintFrame():
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    thisfile = info.filename
    thisfun = info.function
    thisline = info.lineno
    return '{} in {} (@ line {})'.format(thisfile, thisfun, thisline)


def init_net(model, weights, mode, GPU_ID):
    if mode == 0:
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)

    return net


def list_imgs(path='.', ext='jpg'):
    search = os.path.join(path, '*.{}'.format(ext))
    return sorted(glob.glob(search))  # Returns a sorted list


def make_dummy(path, img):
    dummy = np.zeros(shape=img.shape, dtype=np.uint8)
    dname = os.path.join(path, 'dummy.png')
    cv2.imwrite(filename=dname, img=dummy)


def write_list_densedata(source, saveto):
    img_list = sorted(glob.glob(source + "/*.jpg"))
    dummyfile = os.path.join(source, "dummy.png")
    if not os.path.exists(dummyfile):
        img0 = cv2.imread(img_list[0])
        make_dummy(source, img0)

    listfile = os.path.join(saveto, "list.txt")
    with open(listfile, "w") as f:
        for img in img_list:
            txt = '{} {}\n'.format(img, dummyfile)
            f.write(txt)

    return listfile


def substitute_img_list(filename, saveto, listfile, target='PLACEHOLDER'):
    # TODO Add support for placing in BATCHSIZE keyword
    f = open(filename, "r")
    data = f.read()
    f.close()

    newdata = data.replace(target, listfile)  # replace strings
    filebase = os.path.basename(filename)
    writein = os.path.join(saveto, filebase)
    f = open(writein, "w")
    f.write(newdata)
    f.close()

    return writein


def impose_colors(label, colors):
    r = label.copy()
    g = label.copy()
    b = label.copy()

    u_labels = np.unique(label)

    for i, l in enumerate(u_labels):
        bin_l = (
            label == l)  # TODO add here tracking for how many of what class
        r[bin_l] = colors[l, 0]
        g[bin_l] = colors[l, 1]
        b[bin_l] = colors[l, 2]

    #TODO here fix so it just uses one cat to join r, g, and b
    rgb = np.dstack((b,g,r))
    return rgb


def get_output(d, pred, out, colors):
    # TODO Add support for BATCHSIZE > 1
    # UPGRADE all options except "prob" are outdated

    if "result" in d:
        labels = np.argmax(out, axis=0)
        x = impose_colors(labels, colors)

    elif "prob" in d:
        # MAGIC:
        # "prob_X"
        layer = int(d[5])  # This will work as long as"
        # 1. "prob" is in front &&
        # 2. there are only single digit number of classes.
        # TODO replace with regex
        try:
            x = out[layer, :, :] * 255
        except:
            # There is no corresponding output layer
            # TODO how to use warning()
            x = np.zeros(shape=(256,256), dtype=np.float64) + 0.5

    elif d == "label":
        ## Same as probability; might have to add an argument
        # No idea what this was supposed to do . -NI, 3-16-17
        pass

    else:
        x = np.argmax(out, axis=0)

    return x


def process(exphome, expdirs, model_template, weights, mode=1, GPU_ID=0, reportfile='./log.txt'):
    # Force dest to be a list
    start_time = time.time()
    if isinstance(expdirs[1:], basestring):
        expdirs[1:] = [expdirs[1:]]

    repf = open(reportfile, 'a')

    listfile = write_list_densedata(expdirs[0], exphome)
    model = substitute_img_list(model_template, exphome, listfile)

    ## Initialize a network - reference implementation uses pycaffe
    net = init_net(model, weights, mode, GPU_ID)

    imgs = list_imgs(path=expdirs[0])

    # TODO PULL NUMBER  COLORS FROM NET DEF
    colors = generate_color.generate(n=5, whiteidx=3, cmap='jet')

    for i, img in enumerate(imgs):
        if i % 100 == 0:
            print '\tHistoseg processing img {} / {}'.format(i, len(imgs))
            repf.write('\tHistoseg processing img {} / {}\n'.format(i,
                                                                    len(imgs)))

        _ = net.forward()
        pred = net.blobs['prob'].data
        out = np.squeeze(pred)  # i.e first image, if a stack

        ## Iterate through the list of directories and write in appropriate outputs:
        write_name_base = os.path.basename(img).replace(
            '.jpg', '.png')  # Fix magic file types
        # be smarter
        for d in expdirs[1:]:
            write_name = os.path.join(d, write_name_base)
            _, d = os.path.split(d)
            x = get_output(d, pred, out, colors)
            cv2.imwrite(filename=write_name, img=x)

    end_time = time.time()
    elapsed = (end_time - start_time)
    print '\nTIME PROCESSING {} images in {}'.format(len(imgs), elapsed)

    repf.write('TIME INFERENCE {} images in {}\n'.format(len(imgs), elapsed))

    repf.close()

    # Clean pycaffe from the GPU
    # https://github.com/BVLC/caffe/issues/1702
    del net
