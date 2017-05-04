#!/bin/bash

set -e
module load openslide/3.4.1

histoseg=/home/ingn/histo-seg

echo $1

python $histoseg/core/histo_pipeline.py
