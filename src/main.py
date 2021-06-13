#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:51:34 2021
@author: pantoja
"""

from domain import Domain
import numpy as np
from utils import *
import scipy
from camera import Camera
import cv2
import matplotlib.pyplot as plt
from poses_ransac import PosesModel
from ransac import mod_ransac
import copy
from tracker import Tracker
from seq_utils import triangulate_model, input_bundle_adj, update_structure_motion, find_2d3D_correspond, run_pnl
from bundle_adjustment import run_bundle_adjustment


# Path to folder containing input images
#image_path = '../../input/images0/'
image_path = 'images0/'
# Path to save line segments from lsd algorithm
line_image_path = '../../output/lsd/'

#Matrix K with the intrinsic parameters of the camera
K = np.array([[3414.66, 0, 3036.64], [0, 3413.37, 2015.48], [0, 0, 1]])

# Extracting initial features from images necessary to create a domain object. 
# if manual_annotations_flag is true, the features are taken from csv file with this information
points2d, lines2d, views = initial_features(image_path, line_image_path, manual_annotation_flag=False)
#points2d, lines2d, views = initial_features(image_path, line_image_path, manual_annotation_flag=True)

# Creating Domain object. This will contain all the information related with the structure and motion.
domain = Domain(points2d, lines2d, views, K)

# Matching Lines between image pairs based on endpoints sift descriptors assigned
domain.match_lines(plot=True, triplet=False)
# domain.match_lines_manual(plot=True, triplet=False)

# Matching sift kps between image pairs. Not part of the line sfm pipe line
# but used to do verifications
domain.match_kps(plot=False)

# Creating unit spherical representations for domain points and lines
domain.spherical_representation()

# Cluster group of parallel lines following Manhattan world representation
domain.cluster_lines(K, save_flag=True)

# Ransac like model to find optimal relative rotations
n_iter = 5000 #number of iterations equal to number of Rij hypothesys generated
R01_candid = generate_Rij_hyp(domain, view_id=0, n_iter=n_iter) #Relative rotations hypothesis views 0-1
R12_candid = generate_Rij_hyp(domain, view_id=1, n_iter=n_iter) #Relative rotations hypothesis views 1-2
R01_optim = find_R_optim(R01_candid)
R12_optim = find_R_optim(R12_candid)

# Input global rotations for 3 initial views in eq [9] to calculate their translations
R0 = np.eye(3)  # assuming R0 is not rotated.
R1 = R01_optim @ R0
R2 = R12_optim @ R1

#Applying Ransac to find best model P0, P1, P2 (best translations t0,t1,t2)
v0, v1, v2 = 0, 1, 2
model = PosesModel(domain, K, R0, R1, R2, view0=0, view1=1, view2=2, debug=False) #Creates Ransac model for poses
all_data = find_correspondences_3v(domain, v0, v1, v2, n_corr="full") #find 3v correspondences
ransac_fit, num_inl = mod_ransac(all_data, model, 5, 500, 5, debug=True, return_all=True) #runing ransac algorithm

# Assigning cameras to view objects
P0, P1, P2 = ransac_fit
domain.views[0].P = P0
domain.views[1].P = P1
domain.views[2].P = P2

####PLOTING CAMERAS############################
# Ploting Camera centers initial 3 views
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.axis('on')
ax.scatter(P0.c[0], P0.c[1], P0.c[2], s=10, c='m')
ax.scatter(P1.c[0], P1.c[1], P1.c[2], s=10, c='m')
ax.scatter(P2.c[0], P2.c[1], P2.c[2], s=10, c='m')
P_list = [P0, P1, P2]
plot_cams(P_list, fig=fig)
plot_cam(P0, c="R", fig=fig)
plot_cam(P1, c="G", fig=fig)
plot_cam(P2, c="B", fig=fig)


# Computing tracker (arrays with tracks information)
Tr = Tracker() #Create Tracker object
Tr.comp_tracker(domain, last_view=1) #compute tracker information taking as last view registered 1
domain.tracker = Tr #assign track to the domain

# Triangulation from tracks to Lines3D
triangulate_model(domain, last_view=1)
# ploting model before BA
domain.plot_lines3d(color='k')

# Bundle adjustment
input_bundle_adjustment = input_bundle_adj(domain, rotate_model="Rodrigues", last_view=1)
op_camera_params, op_points_3d, residual = run_bundle_adjustment(input_bundle_adjustment, loss_f='linear')

# Update 3D lines domain.lines3D and view cameras domain.views[i].P
update_structure_motion(domain, op_camera_params, op_points_3d, last_view=1)
domain.plot_lines3d(color='b')

################
# Registering new views following pnl approach#########################################
################
for view_added in range(2, len(domain.views)):
    # Find 2d3D correspondances between new view and existant structure based on last view
    line_corr_2d3d, _ = find_2d3D_correspond(domain, view_added)

    # Run pnl to get camera P from new registered view
    Pi = run_pnl(domain, line_corr_2d3d)
    domain.views[view_added].P = Pi

    # TRIANGULATING FOR NEW POSES STATUS AND RUNNING BUNDLE ADJUSTMENT AGAIN !!! (MODIFY JUST TO RUN FOR THE LAST 5 VIEWS)
    # Computing tracker
    Tr = Tracker()
    Tr.comp_tracker(domain, last_view=view_added)
    domain.tracker = Tr

    # Triangulation from tracks to Lines3D
    triangulate_model(domain, last_view=view_added)
    
    # Bundle adjustment
    input_bundle_adjustment = input_bundle_adj(domain, rotate_model="Rodrigues", last_view=view_added)
    op_camera_params, op_points_3d, residual = run_bundle_adjustment(input_bundle_adjustment, loss_f='linear')
    
    # Update 3D lines domain.lines3D and view cameras domain.views[i].P
    update_structure_motion(domain, op_camera_params, op_points_3d, last_view=view_added)

domain.plot_lines3d(color='g')

# PLOTING CAMERAS
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.axis('on')

# Ploting Camera centers
fig = plt.figure()
ax = fig.gca(projection='3d')
P_list = [i.P for i in domain.views]
plot_cams(P_list, fig=fig)



