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

CAFFE_ROOT = '/home/nathan/caffe-segnet-cudnn5'
#CAFFE_ROOT = '/Users/nathaning/caffe-segnet-cudnn5'
sys.path.insert(0, CAFFE_ROOT+"/python") 
import caffe

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
    

def define_colors(n = 4):
    # TODO !!!! INSERT A COLOR-WHEELY TYPE ALGORITHM:
    # Define a set of colors for visualization later
    ## screw it, for now:

    # TODO migrate this to pipeline.py ; pass the color matrix through
    c0 = [245, 32, 35] # red - g3
    c1 = [25, 242, 20] # green - g4
    c2 = [35, 35, 220] # blue - BN
    c3 = [255, 255, 255] # white - ST
    c4 = [210, 150, 20] # white - ST

    label_colors = np.array([c0, c1, c2, c3, c4])

    print ""
    print PrintFrame()
    print "Using colors:"
    print label_colors
    return label_colors

def list_imgs(path = '.', ext = 'jpg'):
    search = os.path.join(path, '*.{}'.format(ext))
    return sorted(glob.glob(search)) # Returns a sorted list
   

def make_dummy(path, img):
    dummy = np.zeros(shape = img.shape, dtype = np.uint8)
    dname = os.path.join(path, 'dummy.png')
    cv2.imwrite(filename = dname, img = dummy)


def write_list_densedata(source, saveto):
    img_list = sorted(glob.glob(source+"/*.jpg"))
    dummyfile = os.path.join(source, "dummy.png")
    if not os.path.exists(dummyfile):
        img0 = cv2.imread(img_list[0])
        make_dummy(source,img0) 

    listfile = os.path.join(saveto, "list.txt")
    with open(listfile, "w") as f:
        for img in img_list:
            txt = '{} {}\n'.format(img, dummyfile)
            f.write(txt)

    return listfile


def substitute_img_list(filename, saveto, listfile, target = 'PLACEHOLDER'):
    # TODO Add support for placing in BATCHSIZE keyword
    f = open(filename, "r")
    data = f.read()
    f.close()

    newdata = data.replace(target, listfile) # replace strings
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
        bin_l = (label == l) # TODO add here tracking for how many of what class
        r[bin_l] = colors[l, 0]
        g[bin_l] = colors[l, 1]
        b[bin_l] = colors[l, 2]

    #TODO here fix so it just uses one cat to join r, g, and b
    rgb = np.zeros(shape = (label.shape[0], label.shape[1], 3), dtype = np.uint8)
    rgb[:,:,2] = r #RGB --> BGR ?? I think OpenCV uses BGR. Ditch OpenCV and use scipy? 
    rgb[:,:,1] = g
    rgb[:,:,0] = b
    return rgb 


    '''
    This is the collection of if's that controls what gets passed back/written out

    These functions are still really pipeliney.

    If I ever want new types of output, this is a good place to start implementing that.

    The keys are passed through the whole pipeline.py as distinct folders
    I could say, I want prob3 - meaning the probability image from class #3 
    That would totally work.

    Figure out a new type of overlay function for the cases like prob, and single class
    labels that make the most sense as 2D arrays; no 3rd dimension (impose_colors won't work). 

    NEW procedure:
    if none of the probs is > a cutoff, pick a default class.--

        unsure = [p0 < cutoff and p1 < cutoff .... pn < cutoff]
        label[unsure] = default
    '''
def get_output(d, pred, out, colors):
    # TODO Add support for BATCHSIZE > 1
    
    #out = np.argmax(out, axis=0) # argmax classifier ; can change
    # if d == 'debug':
    #     x = net.blobs['data'].data
    #     x = np.squeeze(x)
    #     x = np.swapaxes(x, 0, 2)
    #     #x = cv2.imread(img)
    #     x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    
    if "result" in d:
        ## Main result ~ label matrix
        #print " doing result task"
        labels = np.argmax(out, axis = 0)
        x = impose_colors(labels, colors)
    
    elif "prob" in d:
        #print "Working in {}\tTODO add a regex to pull correct layer".format(d)
        #layer = int(d[-1]) # TODO THIS IS SKEETTCCHHHYY -- now its broken
        layer = int(d[4]) # This will work as long as"
                          # 1. "prob" is in front &&
                          # 2. there are only single digit number of classes.         
        # TODO replace with regex
        x = out[layer,:,:]*255 
    
    elif d == "label":
        ## Same as probability; might have to add an argument
        # No idea what this was supposed to do . -NI, 3-16-17
        pass
    
    else:
        x = np.argmax(out, axis = 0)

    return x

# Default to CPU
def process(exphome, expdirs, model_template, weights, mode = 1, GPU_ID = 0):
    
    # Force dest to be a list
    if isinstance(expdirs[1:], basestring):
        expdirs[1:] = [expdirs[1:]]

    listfile = write_list_densedata(expdirs[0], exphome)
    model = substitute_img_list(model_template, exphome, listfile)
  
    net = init_net(model, weights, mode, GPU_ID)

    imgs = list_imgs(path = expdirs[0])


    # TODO PULL NUMBER OF COLORS FROM NET DEF   
    #colors = define_colors(4)
    colors = generate_color.hsv(5)

    # decide what outputs to give from the length of 'dest'
    # 'dest' should always be a list; take care of that later. (TODO)
    # main loop:
    #dest = dest[0]
    for i, img in enumerate(imgs):
        if i % 100 == 0:
            print 'Histoseg processing img {} / {}'.format(i, len(imgs))
        
        # Run the network forward once i.e process one image
        # TODO here convert the proto into an 'input' type and process 
        #   each image after a series of rotations, and average the results. 

        _ = net.forward() 
        pred = net.blobs['prob'].data
        out = np.squeeze(pred) # i.e first image, if a stack  

        ## Iterate through the list of directories and write in appropriate outputs:
        write_name_base = os.path.basename(img).replace('.jpg', '.png') # Fix magic file types
        for d in expdirs[1:]:
            write_name = os.path.join(d, write_name_base)
            #print write_name, 
            _, d = os.path.split(d)
            
            x = get_output(d, pred, out, colors)

            cv2.imwrite(filename = write_name, img = x)
            
         
def process_dev(expdirs):
    imgs = list_imgs(path = expdirs[0])
    for i, img in enumerate(imgs):
        if i % 100 == 0:
            print 'Histo-DEV processing img {} / {}'.format(i, len(imgs))
        devimg = cv2.imread(img) 
        devimg = cv2.cvtColor(devimg, cv2.COLOR_RGB2GRAY) 
        devimg = devimg[:,:,0] # Not sure if the GRAY is 3-channel
        
        write_name_base = os.path.basename(img).replace('.jpg', '.png') # Fix magic file types
        for d in expdirs[1:]:
            write_name = os.path.join(d, write_name_base)
            #print write_name, 
            _, d = os.path.split(d)
            cv2.imwrite(filename = write_name, img = devimg)





