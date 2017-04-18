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

'''
code_class = {
    0:'G3',
    1:'G4',
    2:'BN',
    3:'ST',
    4:'G5'
}

metrics_list = [
    'OverallJaccard',
    'OverallAccuracy',
    'Kappa',
    'G3Jaccard',
    'G4Jaccard',
    'BNJaccard',
    'STJaccard',
    'G5Jaccard',
    'G3Accuracy',
    'G4Accuracy',
    'BNAccuracy',
    'STAccuracy',
    'G5Accuracy',
    'G3f1',
    'G4f1',
    'BNf1',
    'STf1',
    'G5f1',
    'G3mattCoef',
    'G4mattCoef',
    'BNmattCoef',
    'STmattCoef',
    'G5mattCoef',
    'G3precision',
    'G4precision',
    'BNprecision',
    'STprecision',
    'G5precision',
    'EPAccuracy',
    'EPf1',
    'EPprecision',
    'PCaAccuracy',
    'PCaf1',
    'PCaprecision'
]

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

def canc(mask):
    m0 = mask == 0
    m1 = mask == 1
    m4 = mask == 4

    m0.dtype = np.uint8
    m1.dtype = np.uint8
    m4.dtype = np.uint8

    m = m0 + m1 + m4
    return m>0


# Dictionary of lambda functions... hope this works
tests_fn = {
    'OverallJaccard': lambda gt,test: jaccard_similarity_score(gt,test),
    'OverallAccuracy':lambda gt,test: accuracy_score(gt,test),
    'Kappa':          lambda gt,test: cohen_kappa_score(gt,test),
    'G3Jaccard':      lambda gt,test: jaccard_similarity_score(gt==0,test==0),
    'G4Jaccard':      lambda gt,test: jaccard_similarity_score(gt==1,test==1),
    'BNJaccard':      lambda gt,test: jaccard_similarity_score(gt==2,test==2),
    'STJaccard':      lambda gt,test: jaccard_similarity_score(gt==3,test==3),
    'G5Jaccard':      lambda gt,test: jaccard_similarity_score(gt==4,test==4),
    'G3Accuracy':      lambda gt,test: accuracy_score(gt==0,test==0),
    'G4Accuracy':      lambda gt,test: accuracy_score(gt==1,test==1),
    'BNAccuracy':      lambda gt,test: accuracy_score(gt==2,test==2),
    'STAccuracy':      lambda gt,test: accuracy_score(gt==3,test==3),
    'G5Accuracy':      lambda gt,test: accuracy_score(gt==4,test==4),
    'G3f1':           lambda gt,test: f1_score(gt==0,test==0),
    'G4f1':           lambda gt,test: f1_score(gt==1,test==1),
    'BNf1':           lambda gt,test: f1_score(gt==2,test==2),
    'STf1':           lambda gt,test: f1_score(gt==3,test==3),
    'G5f1':           lambda gt,test: f1_score(gt==4,test==4),
    'G3mattCoef':     lambda gt,test: matthews_corrcoef(gt==0,test==0),
    'G4mattCoef':     lambda gt,test: matthews_corrcoef(gt==1,test==1),
    'BNmattCoef':     lambda gt,test: matthews_corrcoef(gt==2,test==2),
    'STmattCoef':     lambda gt,test: matthews_corrcoef(gt==3,test==3),
    'G5mattCoef':     lambda gt,test: matthews_corrcoef(gt==4,test==4),
    'G3precision':     lambda gt,test: precision_score(gt==0,test==0),
    'G4precision':     lambda gt,test: precision_score(gt==1,test==1),
    'BNprecision':     lambda gt,test: precision_score(gt==2,test==2),
    'STprecision':     lambda gt,test: precision_score(gt==3,test==3),
    'G5precision':     lambda gt,test: precision_score(gt==4,test==4),
    'EPAccuracy':     lambda gt,test: accuracy_score(epit(gt),epit(test)),
    'EPf1':           lambda gt,test: f1_score(epit(gt),epit(test)),
    'EPprecision':           lambda gt,test: precision_score(epit(gt),epit(test)),
    'PCaAccuracy':    lambda gt,test: accuracy_score(canc(gt),canc(test)),
    'PCaf1':          lambda gt,test: f1_score(canc(gt),canc(test)),
    'PCaprecision':          lambda gt,test: precision_score(canc(gt),canc(test))
}


# //**************************** START ******************************//
def list_matching(gt_dir, test_dir, test_ext):
    gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    test_search = os.path.join(test_dir, '*.{}'.format(test_ext))
    test_list = sorted(glob.glob(os.path.join(test_search)))

    # Filter by what's in the test_list:
    def check_match(gt, test):
        gt = os.path.basename(gt)
        gt,_ = os.path.splitext(gt)
        return gt in test

    for gt,test in zip(gt_list, test_list):
        if check_match(gt, test):
            continue;
        else:
            return 0

    return gt_list, test_list


def print_composition(gt_img, test_img):
    gt_u = np.unique(gt_img)
    test_u = np.unique(test_img)

    print 'GT: {}\tTEST: {}'.format(gt_u, test_u)


def load_images(gt_pth, test_pth):
    gt_base = os.path.basename(gt_pth)
    test_base = os.path.basename(test_pth)
    print '\n{} / {}'.format(gt_base, test_base)
    gt_img = cv2.imread(gt_pth, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    test_img = cv2.imread(test_pth, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    if gt_img.shape != test_img.shape:
        test_img=cv2.resize(test_img, dsize=(gt_img.shape[:2]))

    gt_img = gt_img.reshape(np.prod(gt_img.shape))
    test_img = test_img.reshape(np.prod(test_img.shape))

    print_composition(gt_img, test_img)

    return gt_img, test_img


def eval_mask_pair(gt_img, test_img, verbose = True):
    metrics = {key:0 for key in metrics_list}

    # For class-wise scores, only evaluate if the class
    # is present in ground truth
    gt_u = np.unique(gt_img)
    in_gt = [code_class[key] for key in gt_u]
    for key in metrics.iterkeys():
        if 'Overall' in key or key == 'Kappa':
            # Always do overall , and the kappa score
            metrics[key] = tests_fn[key](gt_img, test_img)
        elif 'PCa' in key and any([u in ['G3','G4','G5'] for u in in_gt]):
            metrics[key] = tests_fn[key](gt_img, test_img)
        elif 'EP' in key and any([u in ['G3','G4','G5','BN'] for u in in_gt]):
            metrics[key] = tests_fn[key](gt_img, test_img)
        elif any([u in key for u in in_gt]):
            metrics[key] = tests_fn[key](gt_img, test_img)
        else:
            if verbose:
                print 'Skipping {}'.format(key)

            metrics[key] = np.nan

    if verbose:
        print_report(metrics)

    return metrics


def metric_list2dict(metrics):
    # Translate the list of metrics to a dict
    overall_metrics = {key:0 for key in metrics_list}
    for key in overall_metrics.iterkeys():
        m = [gt_test[key] for gt_test in metrics]
        overall_metrics[key] = [np.nanmean(m), np.nanstd(m)]

    return overall_metrics


def eval_metrics(gt_list, test_list):
    overall_metrics = []
    for gt_pth, test_pth in zip(gt_list, test_list):
        gt_img, test_img = load_images(gt_pth, test_pth)
        pair_metrics = eval_mask_pair(gt_img, test_img)
        overall_metrics.append(pair_metrics)

    summary = metric_list2dict(overall_metrics)
    return summary


def save_report(reportfile, metrics):
    # Made obsolete by tee utility
    pass


def print_report(metrics):
    # Impose the same order on everything, always
    for key in metrics_list:
        print '{}: {}'.format(key, metrics[key])


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


def main(gt_dir, test_dir, reportfile=None, test_ext='png', subset=None):
    gt_list, test_list = list_matching(gt_dir, test_dir, test_ext)

    if subset is not None:
        gt_list, test_list = random_subset(gt_list, test_list, pct=subset)

    summary = eval_metrics(gt_list, test_list)

    print '\nOverall averages:'
    print_report(summary)


if __name__ == '__main__':
    # debug arguments default
    #gt_dir = '/Users/nathaning/_projects/semantic-pca/data/seg_0.8.1_val_mask'
    #test_dir = '/Users/nathaning/_projects/semantic-pca/data/seg_0.8.1_test/mask'
    reportfile = None
    #test_ext = 'png'

    # subset is float or None
    subset = None

    # Parse command line arguments
    gt_dir = sys.argv[1]
    test_dir = sys.argv[2]
    test_ext = sys.argv[3]
    if len(sys.argv) == 5:
        print "using 4th arg as subset pct"
        subset = np.float64(sys.argv[4])

    main(gt_dir, test_dir, test_ext=test_ext, subset=subset)

