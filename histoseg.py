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

CAFFE_ROOT = '/Users/nathaning/caffe-segnet-segnet-cleaned'
sys.path.insert(0, CAFFE_ROOT+"/python") 
import caffe

# Define inspection code that spits out the line it's called from (as str)
def PrintFrame():
    callerframercord = inspect.stack()[1] 
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    thisfile = info.filename
    thisfun = info.function
    thisline = info.lineno
    return '{} in {} (@ line {})'.format(thisfile, thisfun, thisline)

def init_net(model, weights, mode):
    if mode == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)

    return net
    

def define_colors(n = 4):
    # TODO !!!! INSERT A COLOR-WHEELY TYPE ALGORITHM:
    # Define a set of colors for visualization later
    ## screw it, for now:
    c1 = [245, 32, 35]
    c2 = [25, 242, 20]
    c3 = [35, 35, 220]
    c4 = [255, 255, 255]

    label_colors = np.array([c1, c2, c3, c4])

    return label_names

def list_imgs(pth = '.', ext = 'jpg'):
    search = os.path.join(pth, '*.{}'.format(ext))
    return sorted(glob.glob(search)) # Returns a sorted list
   

def make_dummy(pth, img):
    dummy = np.zeros(shape = img.shape, dtype = np.uint8)
    dname = os.path.join(pth, 'dummy.png')
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
        r[bin_l] = colors[i, 0]
        g[bin_l] = colors[i, 1]
        b[bin_l] = colors[i, 2]

    #TODO here fix so it just uses one cat to join r, g, and b
    rgb = np.zeros(shape = (label.shape[0], label.shape[1], 3), dtype = np.uint8)
    rgb[:,:,2] = r #RGB --> BGR ?? I think OpenCV uses BGR. Ditch OpenCV and use scipy? 
    rgb[:,:,1] = g
    rgb[:,:,0] = b
    return rgb 

def get_output(d, net):
    pred = net.blobs['prob'].data
    out = np.squeeze(pred[0,:,:,:]) # i.e first image, if a stack
    
    out = np.argmax(out, axis=0) # argmax classifier ; can change
    if d == 'debug':
        x = cv2.imread(img)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    elif d == "result":
        ## Main result ~ label matrix
        x = impose_colors(out, colors)
    elif d == "prob":
        ## Figure out how to deal with which probability image to write in. 
        pass
    elif d == "label":
        ## Same as probability; might have to add an argument
        pass
    return net, out

# Default to CPU
def process(exphome, source, dest, model_template, weights, mode = 1):

    listfile = write_list_densedata(source, exphome)
    model = substitute_img_list(model_template, exphome, listfile)
   
    net = init_net(model, weights, mode)

    imgs = list_imgs(pth = source)

    colors = define_colors(4)

    # decide what outputs to give from the length of 'dest'
    # 'dest' should always be a list; take care of that later. (TODO)
    # main loop:
    #dest = dest[0]
    for i, img in enumerate(imgs):
        if i % 100 == 0:
            print 'Histoseg processing img {} / {}'.format(i, len(imgs))
        
        # Run the network forward once i.e process one image
        net, out = inference(net)
       
        ## Iterate through the list of directories and write in appropriate outputs:
        write_name = os.path.basename(img).replace('.jpg', '.png') # Fix magic file types
        for d in dest:
            write_name = os.path.join(d, write_name)

            x = get_output(d, net)
            ## For debug

            cv2.imwrite(filename = write_name, img = x)
            
         
