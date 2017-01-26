import data
import glob
import os


def make_segmentation_training(src, anno):
    data.multiply_data(src, anno)

if __name__ == "__main__":

    src = "/Users/nathaning/databases/pca/seg_0.1/feat"
    anno = "/Users/nathaning/databases/pca/seg_0.1/anno"

    make_segmentation_training(src, anno)
