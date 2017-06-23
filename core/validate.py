#!/usr/bin/python

'''
Perform validation given a list of tests and list of ground truths

contact:
    ing[dot]nathany[at]gmail[dot]com

Changelog
---------
2017-04-15:
    Initial

2017-04-18:
    Made functional
    Added listing of tests
    Tests are maintained in a modular list at the top
    as globally available functions
    Added checks for class presence in ground truth
    Tests are only performed if the class is present in GT

'''

import os
import glob
import sys
import cv2
import numpy as np

# Use sklearn metrics
# Substitutable for any function that takes (y_true, y_pred) as args.
from sklearn.metrics import (jaccard_similarity_score, accuracy_score,
                             average_precision_score, f1_score,
                             cohen_kappa_score, precision_score,
                             matthews_corrcoef)

'''
Functions & Variables:
    gt_list, test_list = list_matching(gt_dir, test_dir, test_ext)
    gt, test = load_images(gt, test)
    overall_metrics = eval_metrics(gt_list, test_list)
    metrics = eval_mask_pair(gt, test)
    save_report(reportfile, metrics)
    print_report(metrics)
    main(gt_dir, test_dir, reportfile, test_ext='png')


Intended usage is to edit these dictionaries to change behavior downstream

'''
code_class = {
    0:'G3',
    1:'G4',
    2:'BN',
    3:'ST',
    # 4:'G5'
}


# Define two special tests:
def epit(mask):
    m0 = mask == 0
    m1 = mask == 1
    m2 = mask == 2
    m4 = mask == 4

    m0.dtype = np.uint8
    m1.dtype = np.uint8
    m2.dtype = np.uint8
    m4.dtype = np.uint8

    # Return the union
    m = m0 + m1 + m2 + m4
    return m>0
#/end epit

def canc(mask):
    m0 = mask == 0
    m1 = mask == 1
    m4 = mask == 4

    m0.dtype = np.uint8
    m1.dtype = np.uint8
    m4.dtype = np.uint8

    m = m0 + m1 + m4
    return m>0
#/end canc

def mIOU(gt, test):
    gt_u = np.unique(gt)
    test_u = np.unique(test)

    labels = [0,1,2,3]
    J = {0: 0.0,
         1: 0.0,
         2: 0.0,
         3: 0.0}
    for L in labels:
        if L in gt_u and L not in test_u:
            J[L] = 0
        elif L in test_u and L not in gt_u:
            J[L] = 0
        elif L not in test_u and L not in gt_u:
            J[L] = 1
        else:
            J[L] = jaccard_similarity_score(gt==L, test==L)
        #/end if
    #/end for
    Jsum = 0
    for key in J.iterkeys():
        Jsum += J[key]

    return Jsum / len(labels)

#/end mIOU



# Dictionary of lambda functions... hope this works
metrics_list = {
    'OverallJaccard': lambda gt,test: jaccard_similarity_score(gt,test),
    'OverallmIOU':    lambda gt,test: mIOU(gt, test),
    'G3Jaccard':      lambda gt,test: jaccard_similarity_score(gt==0,test==0),
    'G4Jaccard':      lambda gt,test: jaccard_similarity_score(gt==1,test==1),
    'BNJaccard':      lambda gt,test: jaccard_similarity_score(gt==2,test==2),
    'STJaccard':      lambda gt,test: jaccard_similarity_score(gt==3,test==3),
    'G3f1':           lambda gt,test: f1_score(gt==0,test==0),
    'G4f1':           lambda gt,test: f1_score(gt==1,test==1),
    'BNf1':           lambda gt,test: f1_score(gt==2,test==2),
    'STf1':           lambda gt,test: f1_score(gt==3,test==3),
    'EPf1':           lambda gt,test: f1_score(epit(gt),epit(test)),
    'EPJaccard':      lambda gt,test: jaccard_similarity_score(epit(gt),epit(test)),
    'PCaf1':          lambda gt,test: f1_score(canc(gt),canc(test)),
    'PCaJaccard':     lambda gt,test: jaccard_similarity_score(canc(gt),canc(test)),
    # 'PCaprecision':          lambda gt,test: precision_score(canc(gt),canc(test))
}


# //**************************** START ******************************//
def list_matching(gt_dir, test_dir, test_ext):
    gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

    test_search = os.path.join(test_dir, '*.{}'.format(test_ext))
    test_list = sorted(glob.glob(os.path.join(test_search)))

    print 'Ground truth images found: {}'.format(len(gt_list))
    print 'Test images found: {}'.format(len(test_list))

    # Filter by what's in the test_list:
    def check_match(gt, test):
        gt = os.path.basename(gt)
        gt,_ = os.path.splitext(gt)
        return gt in test
    #/end check_match

    # Check length
    if len(gt_list) == 0 or len(test_list) == 0:
        # print 'Ground truth images found: {}'.format(len(gt_list))
        # print 'Test images found: {}'.format(len(test_list))
        return 0

    for gt,test in zip(gt_list, test_list):
        if check_match(gt, test):
            continue;
        else:
            print '\n ************************ \n'
            print 'Caught an error in ground truth or test listing.\n'
            print ' ************************ \n'
            return 0
        #/end if
    #/end for

    return gt_list, test_list
#/end list_matching


def print_composition(gt_img, test_img):
    gt_u = np.unique(gt_img)
    test_u = np.unique(test_img)

    print 'GT: {}\tTEST: {}'.format(gt_u, test_u)
#/end print_composition


def load_images(gt_pth, test_pth, verbose=False):
    gt_base = os.path.basename(gt_pth)
    test_base = os.path.basename(test_pth)

    gt_img = cv2.imread(gt_pth, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    test_img = cv2.imread(test_pth, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    if gt_img.shape != test_img.shape:
        test_img=cv2.resize(test_img, dsize=(gt_img.shape[:2]))
    #/end if

    gt_img = gt_img.reshape(np.prod(gt_img.shape))
    test_img = test_img.reshape(np.prod(test_img.shape))

    if verbose:
        print 'GT: {} \t TEST: {}'.format(gt_base, test_base)
        print_composition(gt_img, test_img)
    #/end if

    return gt_img, test_img


def eval_mask_pair(gt_img, test_img, verbose=True):
    metrics = {key:0.0 for key in sorted(metrics_list.iterkeys())}

    # For class-wise scores, only evaluate if the class is present in ground truth
    gt_u = np.unique(gt_img)
    in_gt = [code_class[key] for key in gt_u]
    for key in sorted(metrics.iterkeys()):
        if 'Overall' in key or key == 'Kappa':
            # Always do overall , and the kappa score
            metrics[key] = metrics_list[key](gt_img, test_img)
        elif 'PCa' in key and any([u in ['G3','G4','G5'] for u in in_gt]):
            metrics[key] = metrics_list[key](gt_img, test_img)
        elif 'EP' in key and any([u in ['G3','G4','G5','BN'] for u in in_gt]):
            metrics[key] = metrics_list[key](gt_img, test_img)
        elif any([u in key for u in in_gt]):
            metrics[key] = metrics_list[key](gt_img, test_img)
        else:
            if verbose:
                print 'Skipping {}'.format(key)
            #/end if
        #/end if
            metrics[key] = np.nan
    #/end for

    if verbose:
        print_report(metrics)

    return metrics
#/end eval_mask_pair


def metric_list2dict(metrics):
    # Translate the list of metrics to a dict
    overall_metrics = {key:0 for key in sorted(metrics_list.iterkeys())}
    for key in sorted(overall_metrics.iterkeys()):
        m = [gt_test[key] for gt_test in metrics]
        overall_metrics[key] = [np.nanmean(m), np.nanstd(m)]
    #/end for

    return overall_metrics
#/end metric_list2dict



def print_report(metrics_summary):
    # Impose the same order on everything, always
    for key in sorted(metrics_list.iterkeys()):
        print '{: <18}: {:1.3f} +/- {:1.2f}'.format(key, metrics_summary[key][0], metrics_summary[key][1])
    #/end for
#/end print_report



def eval_metrics(gt_list, test_list, printfreq=500, verbose=False):
    overall_metrics = []
    for index, (gt_pth, test_pth) in enumerate(zip(gt_list, test_list)):
        gt_img, test_img = load_images(gt_pth, test_pth, verbose=verbose)
        pair_metrics = eval_mask_pair(gt_img, test_img, verbose=verbose)
        overall_metrics.append(pair_metrics)

        if index % printfreq == 0:
            print '-----------------------------------------'
            print 'Summary for {} images'.format(index)
            print_report(metric_list2dict(overall_metrics))
    #/end for

    summary = metric_list2dict(overall_metrics)
    return summary
#/end eval_metrics


def save_report(reportfile, metrics_summary):
    print 'Writing summary to {}'.format(reportfile)
    with open(reportfile, 'w') as f:
        for key in sorted(metrics_list.iterkeys()):
            f.write('{: <18}: {:1.3f} +/- {:1.2f}\n'.format(key, metrics_summary[key][0],
                                           metrics_summary[key][1]))
    pass
#/end save_report


def random_subset(gt_list, test_list, pct):
    print 'Generating a random subset to test'
    n = len(gt_list)
    indices = np.random.permutation(np.arange(n))
    indices = indices[:int(n*pct)]
    gt_list = np.asarray(gt_list)
    test_list = np.asarray(test_list)
    gt_list = list(gt_list[indices])
    test_list = list(test_list[indices])
    print 'Using {} test images'.format(len(gt_list))

    return gt_list, test_list
#/end random_subset


def main(gt_dir, test_dir, reportfile=None, subset=None):
    gt_list, test_list = list_matching(gt_dir, test_dir, 'png')

    if subset is not None:
        print 'Picking subset ({} %)'.format(subset * 100)
        gt_list, test_list = random_subset(gt_list, test_list, pct=subset)
    #/end if

    print 'Testing {} ground truth against {} tests'.format(len(gt_list), len(test_list))
    summary = eval_metrics(gt_list, test_list)

    print '\nOverall averages:'
    print_report(summary)
    save_report(reportfile, summary)
#/end main


'''

Finds the agreement metrics for similarly named label images from 2 folders

Usage:
$ python validate.py [Ground Truth Dir] [Test Image Dir]
$ python validate.py /path/masks /path/masks2

'''
if __name__ == '__main__':
    # reportfile = None

    # subset is float or None
    subset = 1.0

    # Parse command line arguments
    gt_dir = sys.argv[1]
    test_dir = sys.argv[2]
    reportfile = sys.argv[3]

    print 'Evaluating results in {}'.format(test_dir)
    print 'Against ground truth in {}'.format(gt_dir)
    main(gt_dir, test_dir, reportfile=reportfile, subset=subset)
