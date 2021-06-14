import json
from view import View
from point import Point
from line import Line
from domain import Domain
import point
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
import warnings
warnings.filterwarnings("ignore")


#Reading toy example information from json file
image_path = '../input/toy_example1/'
        
#testing reading toy file
with open(image_path+'toy_example1_3Dvision.json', 'r') as fp:
    domain_dict = json.load(fp)

#Creating variables with dictionary information GT
GT_lines2d = domain_dict["lines2d"]
GT_points2d = domain_dict["points2d"]
GT_K = np.array(domain_dict["K"])
GT_lines3d = domain_dict["lines3d"]
GT_points3d = domain_dict["points3d"]
GT_views = domain_dict["views"]



#Starting pipeline of SfM with lines
#Creating list of lines2d, points2d, and views objects to initialize domain
K = GT_K
views = []
points2d = []
lines2d = []
for i in GT_views: 
    #break
    V = View()
    V.id = i
    V.name = 'pic_{}.png'.format(i)
    V.image = cv2.imread(image_path+V.name)
    points2d_v = []
    lines2d_v = []
    for j, l in enumerate(GT_views[i]["lines2d"]):
        #endpoints of a line
        pt0 = Point()
        pt0.id = GT_views[i]["pts2d"][2*j][0]
        pt0.coord = GT_views[i]["pts2d"][2*j][1]
        points2d_v.append(pt0)
        pt1 = Point()
        pt1.id = GT_views[i]["pts2d"][2*j+1][0]
        pt1.coord = GT_views[i]["pts2d"][2*j+1][1]
        points2d_v.append(pt1)
        
        L = Line()
        L.id = l[0]
        L.local_id = j
        L.points = [pt0, pt1]
        L.comp_angle()
        lines2d_v.append(L)
    
    #Adding info to view V
    V.lines = lines2d_v
    V.points = points2d_v
    views.append(V)
    
    #Apending full lines2d and points2d
    points2d += points2d_v
    lines2d += lines2d_v


#Initiating SfM
domain = Domain(points2d, lines2d, views, K)


#Matching lines: Generating domain.line_match "manually"
line_match = {}
#sequential match
for i in range(len(domain.views) - 1):
    tup_views = (int(domain.views[i].id), int(domain.views[i+1].id))
    matches_ids = []
    for j in range(len(domain.views[i].lines)):
        matches_ids.append([domain.views[i].lines[j].id, domain.views[i+1].lines[j].id])
    line_match[domain.views[i].id+domain.views[i+1].id] = [tup_views, matches_ids]
#sequential match skiping 1 view
for i in range(len(domain.views) - 2):
    tup_views = (int(domain.views[i].id), int(domain.views[i+2].id))
    matches_ids = []
    for j in range(len(domain.views[i].lines)):
        matches_ids.append([domain.views[i].lines[j].id, domain.views[i+2].lines[j].id])
    line_match[domain.views[i].id+domain.views[i+2].id] = [tup_views, matches_ids]
domain.line_match = line_match
        

# Creating unit spherical representations for domain points and lines
domain.spherical_representation()

# cluster group of parallel lines
domain.cluster_lines(K, save_flag=True)

# Ransac like model to find optimal relative rotations
print("Computing R-------------")
n_iter = 5000
R01_candid = generate_Rij_hyp(domain, view_id=0, n_iter=n_iter)
R12_candid = generate_Rij_hyp(domain, view_id=1, n_iter=n_iter)
R01_optim = find_R_optim(R01_candid)
R12_optim = find_R_optim(R12_candid)

# Input rotations eq [9]
R0 = np.eye(3)  # assuming R1 is not rotated
R1 = R01_optim @ R0  
R2 = R12_optim @ R1

####Applying Ransac to find best model P0, P1, P2
print("Computing t -------------")
v0, v1, v2 = 0, 1, 2
model = PosesModel(domain, K, R0, R1, R2, view0=0, view1=1, view2=2, debug=False)
all_data = find_correspondences_3v(domain, v0, v1, v2, n_corr="full")
ransac_fit, num_inl = mod_ransac(all_data, model, 5, 500, 5, debug=False,
                                 return_all=True)  

# testing################################
P0, P1, P2 = ransac_fit
domain.views[0].P = P0
domain.views[1].P = P1
domain.views[2].P = P2
rR0, rt0 = relativeCameraMotion(P0, P1)
F0 = F_from_KRt2(K, rR0, rt0)
rR1, rt1 = relativeCameraMotion(P1, P2)
F1 = F_from_KRt2(K, rR1, rt1)
rR2, rt2 = relativeCameraMotion(P2, P0)  # could be the other way around (P0,P2)
F2 = F_from_KRt2(domain.K, rR2, rt2)


# Triangulate points views 0-1###################################
# Gathering kps for each view
kp0 = np.array([pt.coord for pt in domain.views[0].points]).T
kp1 = np.array([pt.coord for pt in domain.views[1].points]).T

# Triangulating points
X = cv2.triangulatePoints(domain.views[0].P.P, domain.views[1].P.P, kp0, kp1)

# Dividing by homogeneus coordinate
X = X[:] / X[3]

# Ploting points
# Box plots to see outilers (plot just 2 values among 25 adn 50% quartile)
q0 = np.quantile(X[0], (0, 0.05, .5, 0.95, 1))
q1 = np.quantile(X[1], (0, 0.05, .5, 0.95, 1))
q2 = np.quantile(X[2], (0, 0.05, .5, 0.95, 1))
# Ploting points without outliers
fig = plt.figure()
inliers_ind = \
np.where((q0[1] < X[0]) & (X[0] < q0[3]) & (q1[1] < X[1]) & (X[1] < q1[3]) & (q2[1] < X[2]) & (X[2] < q2[3]))[0]
ax = fig.gca(projection='3d')
ax.scatter(X[0][inliers_ind], X[1][inliers_ind], X[2][inliers_ind], s=0.1, c='k')
plt.axis('on')

# Ploting Camera centers
ax.scatter(P0.c[0], P0.c[1], P0.c[2], s=10, c='m')
ax.scatter(P1.c[0], P1.c[1], P1.c[2], s=10, c='m')
ax.scatter(P2.c[0], P2.c[1], P2.c[2], s=10, c='m')
####PLOTING CAMERAS############################
P_list = [P0, P1, P2]
plot_cams(P_list, fig=fig)
plot_cam(P0, c="R", fig=fig)
plot_cam(P1, c="G", fig=fig)
plot_cam(P2, c="B", fig=fig)


# Triangulating by duplets for now. Triplets contain just few kps
# Computing tracker
Tr = Tracker()
Tr.comp_tracker(domain, last_view=1)
domain.tracker = Tr

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
#for view_added in range(3, len(domain.views)):
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
    
    update_structure_motion(domain, op_camera_params, op_points_3d, last_view=view_added)
domain.plot_lines3d(color='g')

# PLOTING CAMERAS
fig = plt.figure()
ax = fig.gca(projection='3d')
P_list = [i.P for i in domain.views]
plot_cams(P_list, fig=fig)


#Ground thruth checking##########
#For checking camera centers, the centroid of their coordinates is moved to the
#origin and then normalized (norm=1)
#For checkin camera rotations and translations, they are computed sequentially
P_list_GT = [Camera(np.array(domain_dict["views"][view]["P"])) for view in domain_dict["views"]]
for Pi in P_list_GT:
    Pi.factor()
    Pi.center()
fig = plt.figure()
ax = fig.gca(projection='3d')
plot_cams(P_list_GT, fig=fig, s=1)    
#Check camera centers with ploting#
#SfM
centers = np.array([Pi.c for Pi in P_list])
centers_centroid = np.mean(centers, axis=0)
centers_norm = centers - centers_centroid
centers_norm /= np.linalg.norm(centers_norm, axis=1).reshape((-1,1))
centers_norm = np.round(centers_norm, 6)
plot_3D_pts(centers_norm)
#GT
centers_GT = np.array([Pi.c for Pi in P_list_GT])
centers_centroid_GT = np.mean(centers_GT, axis=0)
centers_norm_GT = centers_GT - centers_centroid_GT
centers_norm_GT /= np.linalg.norm(centers_norm_GT, axis=1).reshape((-1,1))
centers_norm_GT = np.round(centers_norm_GT, 6)
plot_3D_pts(centers_norm_GT)

#Relative translations and rotations SfM
relR = []
relt = []
for i in range(len(P_list)-1):
    rR, rt = relativeCameraMotion(P_list[i],P_list[i+1])
    relR.append(rR)
    relt.append(rt)
#Relative translations and rotations GT
relR_GT = []
relt_GT = []
for i in range(len(P_list)-1):
    rR, rt = relativeCameraMotion(P_list[i],P_list[i+1])
    relR_GT.append(rR)
    relt_GT.append(rt)


#Relative translations and rotations for SfM and GT are equal!
print("Relative rotations for SfM and GT")
print(relR, relR_GT)
print("Relative translations for SfM and GT")
print(relR, relR_GT)