## Histo-Seg
You have found my work-in-progress tool for digital histopathology segmentation & semantic labelling.

The core functions are two-fold:

* Through the [openslide](http://openslide.org) library, provide headache-free processing to reassembly of Gigapixel sized images.

* Image processing & machine vision functions based on state of the art Convolutional Neural Network architectures. 


Segmentation and semantic labelling uses the [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/) architecture [github](https://github.com/alexgkendall/caffe-segnet).
Other methods such as tile-level classification, Fully Connected Networks, and Fully Connected Conditional Random Fields may be substituted for quick comparison.


### Examples use cases 
* Use-case 1: prostate cancer growth patterns, from manual annotation

* Use-case 2: clear cell renal cancer microenviornment, from automatic Immunohistochemistry annotation

* Use-case 3: lung adenocarcinoma growth patterns, with converted Full-Connected deconvolution layers; trained on whole tile examples.


### Preparing data
Training data follows the "data - label" pair model. Each "data" image should be accompanied by a similarly sized "label" image indicating ground truth examples for the classification. The annotations often indicate discrete classes canonically defined by pathologist consensus.

In histopathology, training data must be curated with the domain knowledge of a trained pathologist. Annotation scarcity is a well documented shortcoming in the field (citations), and represents a significant bottleneck in training data-driven models. Therefore, it's common to use data augmentation pre-processing steps which effectively multiply the area used for training. Some data augmentation implemented here includes:
* Random sub-sampling at variable scales
* Color augmentation in LAB space 
* Image rotation


### Processing Methodology
After a segmentation model is trained, the most interesting application setting is to whole mount slides or biopsies. These are the smallest unit of tissue that pathologists evaluate for the presence and severity of diseased cells. A major aim of this package is to emulate a pathologist's evalutation. 
 
Processing happens in 3 phases:
* Data preparation from Whole Slide Images (WSI) and low-level ROI finding
* High-resolution discretized processing
* Process results agglomeration and report generation

These phases are implemented as individual packages, together composing the "core" module. Since each phase depends only on the previous phase being completed, they are executable in isolation for fast idea prototyping. For example, I have 10 slides to process. Phase 1 performs the basic tissue-finding and tiling well. There is no longer much need to performe phase 1 if I want to try options in phases 2 and 3. 

It was a side goal to allow execution on a massively parallel enviornment like an HPC cluster for discrete phases. Then to pull the results into a central system for further processing, analysis, and long term storage. This goal has yet to be realized. 


Questions & comments to ing[dot]nathany[at]gmail[dot]com
