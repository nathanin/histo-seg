#!/bin/bash
set -e

svsdir=/home/nathan/data/pca_wsi
histoseg=/home/nathan/histo-seg/code
# project=/home/nathan/histo-seg/pca/seg_0.8.1024_resume
# weights=/home/nathan/semantic-pca/weights/seg_0.8.1024_resume/norm_iter_15000.caffemodel
# modelproto=/home/nathan/histo-seg/code/segnet_basic_inference.prototxt

# echo "Running tile procedure"
# ls $svsdir/*.svs | parallel -j 4 $histoseg/tile_parallel.py \
#     $project

# echo "Histo-seg. processing"
# ls $svsdir/*.svs | parallel -j 4 ./run.py \
#     $project \
#     $weights \
#     $modelproto

# echo "Running slide reassembly / multiscale aggregation"
# ls $svsdir/*.svs | parallel -j 2 $histoseg/reassemble.py \
#     $project

#for svs in $(ls $svsdir/*.svs); do
#    echo $svs
#    $histoseg/reassemble.py $svs
#done

#echo "Transferring results to Dropbox"
#python ./transfer_results.py

#ls $svsdir/*.svs | parallel -j 4 python $histoseg/histo_pipeline.py


python ./transfer_results.py
