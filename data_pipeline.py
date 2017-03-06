import data
import glob
import os

# /home/nathan/histo-seg/code/data_pipeline.py
# def make_classification_training(src):
#     data.multiply_one_folder(src);

def make_segmentation_training(src, anno):
    data.multiply_data(src, anno)

if __name__ == "__main__":

   # src = "/Users/nathaning/databases/pca/seg_0.3/feat"
   # anno = "/Users/nathaning/databases/pca/seg_0.3/anno_png"

   src = "/home/nathan/semantic-pca/data/seg_0.3/feat"
   anno = "/home/nathan/semantic-pca/data/seg_0.3/anno_png"

   make_segmentation_training(src, anno)
