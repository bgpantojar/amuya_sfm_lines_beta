# amuya_sfm_lines_beta
This repository contains an implementations of the line-based SfM proposed in the paper ``"Structure from Motion with Line Segments Under Relaxed Endpoint Constraints"`` by Branislav Micusik & Horst Wildenauer. 
Link to the paper: [Click here] (https://ieeexplore.ieee.org/document/7035804)

# How to use it?
**1.Clone repository.**
Inside the repository there are three examples. main.py, corresponds to real images. main_toy_example1.py and main_toy_example2.py correspond to toy examples with information created before hand. All the information necessary to run those files is inside the repository.
**2. Create a conda environment and install python packages

``conda create -n vis3d python=3.7.10``

``pip install -r requirements.txt``
**3. Run examples

Open terminal in repository folder and run:
python main.py

*To use with your data, follow the template of main.py*
