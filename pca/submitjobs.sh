#!/bin/bash

set -e

svsdir="/home/ingn/wsi/pca"

for svs in $(ls $svsdir/*.svs); do
    qsub -V ./job.sh $svs  
done

