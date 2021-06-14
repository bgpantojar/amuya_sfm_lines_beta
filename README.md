# amuya_sfm_lines_beta
This repository contains an implementations of the line-based SfM pipeline proposed in the paper **"Structure from Motion with Line Segments Under Relaxed Endpoint Constraints"** by Branislav Micusik & Horst Wildenauer. 

Link to the paper: [Click here](https://ieeexplore.ieee.org/document/7035804)

Please note that this repository does not include all of the steps mentioned in the paper and some of the steps have also been modified. 

# How to use it?
## 1. Clone repository

All necessary data and codes are inside the ``src`` directory. 

## 2. Create a conda environment and install python packages

``conda create -n vis3d python=3.7.10``

``pip install -r requirements.txt``

## 3. Run examples

There are the three following examples:
  1. ``main.py``: It implements the general pipeline to perform line-based SfM on natural images. 
  2. ``main_toy_example1.py`` and ``main_toy_example2.py``: To test our pipeline, we created these two toy examples. 

*To test on your data, follow the template of main.py*

## 4. How to cite us

If you find this implementation useful, please cite us as:

[![DOI](https://zenodo.org/badge/376598498.svg)](https://zenodo.org/badge/latestdoi/376598498)
