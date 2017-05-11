#!/bin/bash

set -e

svsdir="/home/ingn/wsi/pca"

for svs in $(ls $svsdir/*.svs); do
<<<<<<< HEAD
    qsub -V -cwd -N (basename $svs) ./job.sh $svs  
=======
    qsub -V -cwd -N slide-$(basename $svs) ./job.sh $svs  
>>>>>>> ab6e45ab070547773acd7114d8e59c6747c0d364
done

