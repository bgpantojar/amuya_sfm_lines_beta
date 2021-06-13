#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:10:02 2021
@author: pantoja
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from point import Point3D
from line import Line3D
from camera import Camera
from cvxpnpl import pnl


def triangulate_model2(domain, views2triang): 
    """
    Description
    -----
    Triangulates lines to 3D
    Finds a list of Line3D correspondant to the triangulation of the trackers. Tuples need to be sequential eg, [(0,1), (1,2)]
    views2triang: list of tuples with views to triangulate

    Parameters
    ----------
    domain : Domain
        Domain object with information of SfM.
    views2triang : List
        List of tuples with pairs of views to triangulate.

    Returns
    -------
    None.

    """



    tr_counter = len(domain.lines3d)  # counter that helps selection of new lines to be triangulated

    # defining lines3D and points3D starting ids according last id used in lists lines3D and points3D from domain
    if len(domain.lines3d) == 0:
        l3D_id = 0
        p3D_id = 0
    else:
        l3D_id = domain.lines3d[-1].id + 1
        p3D_id = domain.points3d[-1].id + 1
    # for v in range(len(domain.views)-1):
    for v in views2triang:
        P0 = domain.views[v[0]].P.P
        P1 = domain.views[v[1]].P.P

        nl2tri_it = domain.tracker.n_new_tracks[v[0]]  # number lines to triang in the it
        lines2triang_it = domain.tracker.ids[tr_counter:tr_counter + nl2tri_it, v[0]:v[0] + 2]
        for l in lines2triang_it:
            pts_v0 = np.array([p.coord for p in domain.lines2d[int(l[0])].points],
                              dtype=np.float32).T  # end points view0
            pts_v1 = np.array([p.coord for p in domain.lines2d[int(l[1])].points],
                              dtype=np.float32).T  # end points view1

            # Triangulating points
            X = cv2.triangulatePoints(P0, P1, pts_v0, pts_v1)
            # Dividing by homogeneus coordinate
            X = X[:3] / X[3]
            # Adding points3D to list
            pt3D_ini = Point3D(p3D_id, X[:, 0])
            domain.points3d.append(pt3D_ini)
            p3D_id += 1
            pt3D_fin = Point3D(p3D_id, X[:, 1])
            domain.points3d.append(pt3D_fin)
            p3D_id += 1
            # Adding points3D lines3D to list
            ln3D = Line3D(l3D_id, [pt3D_ini, pt3D_fin])
            l3D_id += 1
            domain.lines3d.append(ln3D)

        tr_counter += nl2tri_it  # updating counter


def triangulate_model(domain, last_view):  
    """
    Description
    ------
    Triangulates lines to 3D
    Finds a list of Line3D correspondant to the triangulation of the trackers
    until last view wished to be registered.     

    Parameters
    ----------
    domain : Domain
        Domain object with information of SfM.
    last_view : int
        Last view registered.

    Returns
    -------
    None.

    """
    


    l3D_id = 0
    p3D_id = 0

    tr_counter = 0
    domain.lines3d = []
    domain.points3d = []

    # for v in range(len(domain.views)-1):
    for i in range(last_view):
        P0 = domain.views[i].P.P
        P1 = domain.views[i + 1].P.P
        nl2tri_it = domain.tracker.n_new_tracks[i]

        lines2triang_it = domain.tracker.ids[tr_counter:tr_counter + nl2tri_it, i:i + 2]

        for l in lines2triang_it:
            pts_v0 = np.array([p.coord for p in domain.lines2d[int(l[0])].points],
                              dtype=np.float32).T  # end points view0
            pts_v1 = np.array([p.coord for p in domain.lines2d[int(l[1])].points],
                              dtype=np.float32).T  # end points view1

            # Triangulating points
            X = cv2.triangulatePoints(P0, P1, pts_v0, pts_v1)
            # Dividing by homogeneus coordinate
            X = X[:3] / X[3]
            # Adding points3D to list
            pt3D_ini = Point3D(p3D_id, X[:, 0])
            domain.points3d.append(pt3D_ini)
            p3D_id += 1
            pt3D_fin = Point3D(p3D_id, X[:, 1])
            domain.points3d.append(pt3D_fin)
            p3D_id += 1
            # Adding points3D lines3D to list
            ln3D = Line3D(l3D_id, [pt3D_ini, pt3D_fin])
            l3D_id += 1
            domain.lines3d.append(ln3D)

        tr_counter += nl2tri_it  # updating counter


def input_bundle_adj(domain, rotate_model="Rodrigues", last_view=2):  
    """
    Descrition
    --------
    # Creates the input necessary to do bundle adjustment.
    # last_view: to define the last view to be taken into account in the BA
    # NECESSARY TO PUT INNITIAL VIEW ALSO. THEN IT IS POSSIBLE DEFINE WHAT ARE
    # THEO POSES TO BE UPDATED DURING BA. DO IT LATER -> NEED TO CHANGE ALSO UPDATE STRUCTURE MOTION FUNCTION.
    # Needs later to assign the views to be applied BA
   
    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    rotate_model : str, optional
        How to express rotation inside bundle adjustment. The default is "Rodrigues".
    last_view : int, optional
        What is the last view wished to be included during BA. The default is 2.

    Returns
    -------
    input_bundle_adjustment : dict
        Dictionary with information necessary for bundle adjustment.

    """    


    # Variables to be input in bundle adjustment
    if rotate_model == "Rodrigues":
        P_i = np.empty(shape=[0, 6])  # cameras parameters (3 rotation parameters + 3 translation)
    else:
        P_i = np.empty(shape=[0, 12])  # cameras parameters

    # Camera params
    for i, v in enumerate(domain.views):

        if i == last_view + 1: break  #

        c_i_it = v.P.c.reshape((3, 1))  # as c is independent of R.
        R_i_it = v.P.R
        if rotate_model == "Rodrigues":
            R_i_it = cv2.Rodrigues(R_i_it)[0].reshape((3, 1))
            P_i_it = np.concatenate((R_i_it, c_i_it), axis=1)  # 3 x 2 matrix
            P_i_it = P_i_it.reshape((1, 6))
        else:
            P_i_it = np.concatenate((R_i_it, c_i_it), axis=1)
            P_i_it = P_i_it.reshape((1, 12))
        P_i = np.concatenate((P_i, np.copy(P_i_it)))
    # P_i will have the shape of (3, 6) => 3 cameras and for each camera 6 params
    # Image points with camera and track id
    check = np.sum(domain.tracker.bin[:, :domain.tracker.bin.shape[1]], axis=1)  # how many times each track is seen
    cond1 = np.where(check > 1)[0]  # Check if track is in at least 2 views
    ids = np.where(domain.tracker.bin[:len(cond1), :domain.tracker.bin.shape[1]] == 1)
    # xi = kps_tracks['Track_pts'][:, :len(geo_nv_dict['Motion'])+1][ids]
    line_i = domain.tracker.ids[:, :domain.tracker.bin.shape[1] + 1][ids]
    line_i = [domain.lines2d[int(i)] for i in line_i]
    line_i_P_id = ids[1]  # id of camera that sees 3D line i
    line_i_3D_id = ids[0]  # id of the 3D line seen by a view as line i

    # 3D triangulated lines from tracks
    L_i = domain.lines3d  # NEED TO BE CHANGED IN CASE OF CONSIDERING JUST SOME VIEWS

    input_bundle_adjustment = {}
    input_bundle_adjustment['camera_params'] = P_i
    input_bundle_adjustment['Lines_3d'] = L_i
    input_bundle_adjustment['camera_indices'] = line_i_P_id.astype('int')
    input_bundle_adjustment['lines_indices'] = line_i_3D_id.astype('int')
    # input_bundle_adjustment['lines_2d_id'] = line_i
    input_bundle_adjustment['lines_2d'] = line_i

    return input_bundle_adjustment


def update_structure_motion(domain, op_camera_params, op_points_3d, last_view=2):
    """
    Description
    --------
    # Updates poses and 3D lines from domain. So far this takes into account for BA all the views
    # triangulated. Change later to take into account just some views! (eg last 5 views for instance)

    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    op_camera_params : numpy.ndarray
        Camera parameters given by BA.
    op_points_3d : numpy.ndarray
        Array with 3D line segments end points of Structure given by BA.
    last_view : int, optional
        id last view wished to be updated. The default is 2.

    Returns
    -------
    None.

    """
    

    for i, v in enumerate(domain.views):
        if i == last_view + 1: break
        cam_par = op_camera_params[i]
        cam_par = cam_par.reshape((3, 2))
        R = cam_par[:, 0]
        R = np.array(cv2.Rodrigues(R)[0])
        c = cam_par[:, 1]
        t = np.array([np.dot(R, -c)]).reshape((3, 1))
        Pi = Camera(domain.K @ np.concatenate((R, t), axis=1))
        Pi.factor()
        Pi.center()
        # domain.views[i].P = Pi
        v.P = Pi

    for i, L in enumerate(domain.lines3d):
        L.points[0].coord = op_points_3d[2 * i]
        L.points[1].coord = op_points_3d[2 * i + 1]



def find_2d3D_correspond(domain, view_added):
    """
    Description
    -------
    Finding 2d-3D correspondances with last view added

    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    view_added : int
        id view to be registered.

    Returns
    -------
    line_corr_2d3d : numpy.ndarray
        Correspondences 3D-2D found for view to be registered in relation with 
        existent 3D structure.
    line_corr_2d2d_for_struc: numpy.ndarray
        Correspondences 2D-2D found for view to be registered in relation with 
        existent 3D structure.
    

    """
    # view_added = 3

    line_corr_2d2d = np.array(domain.line_match["{}{}".format(view_added - 1, view_added)][1])
    tracks_last_view = domain.tracker.ids[domain.tracker.bin[:, view_added - 1] == 1, view_added - 1]
    # Which line correspondances have gotten tracks
    line_corr_2d2d_with_tracks = np.array([[i, j[1]] for i in tracks_last_view for j in line_corr_2d2d if i == j[0]])
    # What are the ids of the 3d lines that correspond to the 2d in the added view
    line_corr_3d = np.array(
        [np.where(i == domain.tracker.ids[:, view_added - 1])[0][0] for i in line_corr_2d2d_with_tracks[:, 0]]).reshape(
        (-1, 1))
    # line correspondaces 2d to 3d for the new added view and existant structure
    line_corr_2d3d = np.concatenate((line_corr_2d2d_with_tracks[:, 1].reshape((-1, 1)), line_corr_3d), axis=1)
    # which are the correspondances 2d2d that do not have tracks so far and will be the additional lines to the structure
    # after registering new view
    line_corr_2d2d_for_struc = np.array([line_corr_2d2d[i] for i in range(len(line_corr_2d2d)) if
                                         line_corr_2d2d[i, 0] not in line_corr_2d2d_with_tracks[:, 0]])

    return line_corr_2d3d, line_corr_2d2d_for_struc  # check the ...for_str.. dim of corr2d2d should be = dim corrtracks+coorstruc


def run_pnl(domain, line_corr_2d3d):
    """
    Description
    ------
    Uses perspective n lines to find pose of new view registered using 
    3D-2D correspondences

    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    line_corr_2d3d : numpy.ndarray
        Correspondences 3D-2D found for view to be registered in relation with 
        existent 3D structure.

    Returns
    -------
    P : Camera
        Camera object of new view registered.

    """
    line_3d = np.array([[domain.lines3d[int(i)].points[0].coord, domain.lines3d[int(i)].points[1].coord] for i in
                        line_corr_2d3d[:, 1]])
    line_2d = np.array([[domain.lines2d[int(i)].points[0].coord, domain.lines2d[int(i)].points[1].coord] for i in
                        line_corr_2d3d[:, 0]])
    poses = pnl(line_2d=line_2d, line_3d=line_3d, K=domain.K)
    R, t = poses[0]
    P = Camera(domain.K @ np.concatenate((R, t.reshape((3, -1))), axis=1))
    P.factor()
    P.center()

    return P