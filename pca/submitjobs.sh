#!/bin/bash

set -e

svsdir="/home/ingn/wsi/pca"

for svs in $(ls $svsdir/*.svs); do
    qsub -V -cwd -N (basename $svs) ./job.sh $svs  
done

