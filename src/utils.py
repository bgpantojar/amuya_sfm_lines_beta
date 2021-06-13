#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:38:12 2021
@author: pantoja
"""

import glob
import os
from view import View
from point import Point
from line import Line
from line_detector import lsd
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import sympy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy
import cv2
from camera import Camera
from tqdm import tqdm
from scipy.spatial.distance import cdist
import json
import pandas as pd


def initial_features(image_path, line_image_path, manual_annotation_flag=False):
    """
    Description
    -------
    Take input images, extract line primitives with lsd and return objects' lists
    from the class lines, points and views to generate afterward a Domain object.

    Parameters
    ----------
    image_path : str
        Path to the folder containing imput images.
    line_image_path : str
        Path where lsd outputs are saved.
    manual_annotation_flag : bool, optional
        If True, it will read the initial features from a csv file containing 
        the manual anotations. The default is False.

    Returns
    -------
    points2d : list
        List with objects from the class Point correspondent to the line
        segments endpoints.
    lines2d : list
        List with objects from the class Line correspondent to the line 
        segments detected from lsd.
    views : List with objects from the class view correspondent to each input
        image.
    

       

    """

    full_image_names = [os.path.split(i)[-1] for i in glob.glob(os.path.join(image_path, '*.JPG'))]
    full_image_names += [os.path.split(i)[-1] for i in glob.glob(os.path.join(image_path, '*.png'))]
    full_image_names += [os.path.split(i)[-1] for i in glob.glob(os.path.join(image_path, '*.jpg'))]
    full_image_names.sort()
    print("Input images: ", full_image_names)
    views = []
    pts_id = 0
    l_id = 0
    lines2d = []
    points2d = []
    for v_id, full_image_name in enumerate(full_image_names):
        V = View()
        V.id = v_id
        V.name = full_image_name
        print('Detecting lines on images ...')
        V.image, lines_coord, _ = lsd(full_image_name, image_path, line_image_path)
        if manual_annotation_flag:
            #df = pd.read_csv(os.path.join('../../manual_line_annotation', full_image_name[:-4]+'_csv.csv'))
            df = pd.read_csv(os.path.join('manual_line_annotation', full_image_name[:-4]+'_csv.csv'))
            lines_coord = []
            for ind in [*range(df.index.stop)]:
                dict = json.loads(df['region_shape_attributes'][ind])
                x_coords = dict['all_points_x']
                y_coords = dict['all_points_y']
                if len(x_coords) != 2 or len(y_coords) != 2: # because of wrong annot
                    x_coords = [x_coords[0], x_coords[-1]]
                    x_coords = [y_coords[0], y_coords[-1]]
                lines_coord.append([x_coords[0], y_coords[0], x_coords[1], y_coords[1]])
        lines2d_v = []
        points2d_v = []
        for local_id, l in enumerate(lines_coord):
            # Point objects that define line
            P1 = Point()
            P1.id = pts_id
            P1.coord = l[:2]
            pts_id += 1
            P2 = Point()
            P2.id = pts_id
            P2.coord = l[2:4]
            pts_id += 1
            # Defining line object
            L = Line()
            L.id = l_id
            l_id += 1
            L.local_id = local_id
            L.points = [P1, P2]
            L.comp_angle()

            # Appending line and point objects to view lists
            lines2d_v.append(L)
            points2d_v.append(P1)
            points2d_v.append(P2)

        # Assigning lines and points lists to view
        V.lines = lines2d_v
        V.points = points2d_v

        # Adding lines2d_v, points2d_v to lines2d, points2d, for domain
        lines2d = lines2d + lines2d_v
        points2d = points2d + points2d_v

        # Finding descriptors for points in view
        V.find_line_desc()
        V.find_kps()
        views.append(V)

    return points2d, lines2d, views


def compute_Rj(lines2d, id_line_1, id_line_2, id_line_3):
    """
    Description
    --------
    Finds the view rotation in relation with vanishing points directions
    represented by 2 parallel lines and one perpendicular to them in 3D
    Follows eq. 4 paper

    Parameters
    ----------
    lines2d : list
        List of Line objects.
    id_line_1 : int
        id for line 1.
    id_line_2 : int
        id for line 2.
    id_line_3 : int
        id for line 3.

    Returns
    -------
    Rj_l : numpy.ndarray
        Rotation matrix in relation with vanishing directions.

    """


    n_1 = lines2d[id_line_1].n  # .reshape((3,1))
    n_2 = lines2d[id_line_2].n  # .reshape((3,1))
    n_3 = lines2d[id_line_3].n  # .reshape((3,1))
    d_1_a = np.cross(n_1, n_2)
    d_1_b = -np.cross(n_1, n_2)
    d_2_a = np.cross(d_1_a, n_3)
    d_2_b = np.cross(d_1_b, n_3)

    d_1 = [d_1_a, d_1_b]
    d_2 = [d_2_a, d_2_b]

    Rj_l = []
    for i in range(len(d_1)):
        for j in range(len(d_2)):
            Rj_col_1 = (d_1[i] / np.linalg.norm(d_1[i])).reshape((3, 1))
            Rj_col_2 = (d_2[j] / np.linalg.norm(d_2[j])).reshape((3, 1))
            Rj_col_3 = ((np.cross(d_1[i], d_2[j])) / np.linalg.norm(np.cross(d_1[i], d_2[j]))).reshape((3, 1))
            Rj = np.hstack((Rj_col_1, Rj_col_2, Rj_col_3))
            Rj_l.append(Rj)
    return Rj_l


def is_rotation_matrix(R):
    """
    Description
    -------
    Checks if a matrix is a valid rotation matrix.

    this function is taken from https://learnopencv.com/rotation-matrix-to-euler-angles/
    

    Parameters
    ----------
    R : numpy.ndarray
        Rotation matrix.

    Returns
    -------
    bool
        Boolean that determines if its or not rotation matrix.

    """
    
    Rt = np.transpose(R)
    must_be_identity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - must_be_identity)
    return n < 1e-6


def rotation_matrix_as_euler_angles(R):
    """
    Description
    -------
    Computes representation of rotation matrix R as euler angles

    Parameters
    ----------
    R : numpy.ndarray
        Rotation matrix.

    Returns
    -------
    euler_angles : numpy.ndarray
        3 euler angles to represent rotation matrix.

    """
    r = Rotation.from_matrix(R)
    euler_angles = r.as_euler('zyx', degrees=True)
    return euler_angles


def create_matrix_A(R1, R2, R3, n_1_1, n_1_2, n_1_3, n_1_4, n_1_5,
                    n_2_1, n_2_2, n_2_3, n_2_4, n_2_5,
                    n_3_1, n_3_2, n_3_3, n_3_4, n_3_5,
                    p_1_1, p_1_2, p_1_3, p_1_4, p_1_5):
    
    """
    Description
    -------
    Approach2 (AR)
    Computes matrix A form equation (9). It's nule space will contain the 
    information of camera translation t1,t2 and one 3D endpoint for 
    each line correspondence of the 5 used for computing A

    Parameters
    ----------
    domain : Domain
        Domain object with informations of SfM.
    R0 : numpy.ndarray
        Rotation matrix view 0.
    R1 : numpy.ndarray
        Rotation matrix view 1.
    R2 : numpy.ndarray
        Rotation matrix view 2.
    corr_3v : numpy.ndarray
        5 lines correspondences ids in 3 views.

    Returns
    -------
    AA : numpy.ndarray
        Matrix A (30x21) from linear equation (9).

    """
    
    A = np.zeros((30, 21), dtype=np.float64)
    #A = np.zeros((20, 21), dtype=np.float64)
    # first equation for 5 lines
    A[0, 0:3] = n_1_1.T @ R1
    A[1, 3:6] = n_1_2.T @ R1
    A[2, 6:9] = n_1_3.T @ R1
    A[3, 9:12] = n_1_4.T @ R1
    A[4, 12:15] = n_1_5.T @ R1

    # second equation for 5 lines
    A[5, 0:3], A[5, 15:18] = n_2_1.T @ R2, n_2_1.T
    A[6, 3:6], A[6, 15:18] = n_2_2.T @ R2, n_2_2.T
    A[7, 6:9], A[7, 15:18] = n_2_3.T @ R2, n_2_3.T
    A[8, 9:12], A[8, 15:18] = n_2_4.T @ R2, n_2_4.T
    A[9, 12:15], A[9, 15:18] = n_2_5.T @ R2, n_2_5.T

    # third equation for 5 lines
    A[10, 0:3], A[10, 18:21] = n_3_1.T @ R3, n_3_1.T
    A[11, 3:6], A[11, 18:21] = n_3_2.T @ R3, n_3_2.T
    A[12, 6:9], A[12, 18:21] = n_3_3.T @ R3, n_3_3.T
    A[13, 9:12], A[13, 18:21] = n_3_4.T @ R3, n_3_4.T
    A[14, 12:15], A[14, 18:21] = n_3_5.T @ R3, n_3_5.T

    # forth set of equations for 5 lines
    A[15, 0:3] = np.array([0, -p_1_1.T @ R1[:, 2], p_1_1.T @ R1[:, 1]], dtype=np.float64)
    A[16, 3:6] = np.array([0, -p_1_2.T @ R1[:, 2], p_1_2.T @ R1[:, 1]], dtype=np.float64)
    A[17, 6:9] = np.array([0, -p_1_3.T @ R1[:, 2], p_1_3.T @ R1[:, 1]], dtype=np.float64)
    A[18, 9:12] = np.array([0, -p_1_4.T @ R1[:, 2], p_1_4.T @ R1[:, 1]], dtype=np.float64)
    A[19, 12:15] = np.array([0, -p_1_5.T @ R1[:, 2], p_1_5.T @ R1[:, 1]], dtype=np.float64)

    A[20, 0:3] = np.array([p_1_1.T @ R1[:, 2], 0, - p_1_1.T @ R1[:, 0]], dtype=np.float64)
    A[21, 3:6] = np.array([p_1_2.T @ R1[:, 2], 0, - p_1_2.T @ R1[:, 0]], dtype=np.float64)
    A[22, 6:9] = np.array([p_1_3.T @ R1[:, 2], 0, - p_1_3.T @ R1[:, 0]], dtype=np.float64)
    A[23, 9:12] = np.array([p_1_4.T @ R1[:, 2], 0, - p_1_4.T @ R1[:, 0]], dtype=np.float64)
    A[24, 12:15] = np.array([p_1_5.T @ R1[:, 2], 0, - p_1_5.T @ R1[:, 0]], dtype=np.float64)

    A[25, 0:3] = np.array([-p_1_1.T @ R1[:, 1], p_1_1.T @ R1[:, 0], 0], dtype=np.float64)
    A[26, 3:6] = np.array([-p_1_2.T @ R1[:, 1], p_1_2.T @ R1[:, 0], 0], dtype=np.float64)
    A[27, 6:9] = np.array([-p_1_3.T @ R1[:, 1], p_1_3.T @ R1[:, 0], 0], dtype=np.float64)
    A[28, 9:12] = np.array([-p_1_4.T @ R1[:, 1], p_1_4.T @ R1[:, 0], 0], dtype=np.float64)
    A[29, 12:15] = np.array([-p_1_5.T @ R1[:, 1], p_1_5.T @ R1[:, 0], 0], dtype=np.float64)

    return A


def is_lin_ind(v1,v2, verbose=False):
    """
    Description
    -------
    To determine if 2 vectors are LI or LD

    Parameters
    ----------
    v1 : numpy.ndarray
        vector 1.
    v2 : numpy.ndarray
        vector 2.
    verbose : bool, optional
        Print verbose. The default is False.

    Returns
    -------
    str
        LI or LD according case.

    """
            
    
    matrix = np.array([v1, v2])
    
    reduced, indexes = sympy.Matrix(matrix).T.rref()  # T is for transpose
    
    if verbose:
        print(reduced)
        print(indexes)
        print(matrix[indexes,:])
    
    if len(indexes) == 2:
        if verbose: print("linearly independant")
        return "LI"
    else:
        if verbose: print("linearly dependant")
        return "LD"

def find_matrix_A(domain, R0, R1, R2, corr_3v):
    """
    Description
    -------
    Computes matrix A form equation (9). It's nule space will contain the 
    information of camera translation t1,t2 and one 3D endpoint for 
    each line correspondence of the 5 used for computing A

    Parameters
    ----------
    domain : Domain
        Domain object with informations of SfM.
    R0 : numpy.ndarray
        Rotation matrix view 0.
    R1 : numpy.ndarray
        Rotation matrix view 1.
    R2 : numpy.ndarray
        Rotation matrix view 2.
    corr_3v : numpy.ndarray
        5 lines correspondences ids in 3 views.

    Returns
    -------
    AA : numpy.ndarray
        Matrix A (30x21) from linear equation (9).

    """
    # Creates matrix A.
    #a1, a2, a3 : first, second and third equations for 5 points
    #b1, b2, b3: equations from fourth equation (cross product) for 5 points
    
    #finding n1, n2, n3 for each line in correspondances
    n0 = []
    n1 = []
    n2 = []
    p0 = []
    p0_e = []
    for i in corr_3v:
        n0.append(domain.lines2d[i[0]].n)
        n1.append(domain.lines2d[i[1]].n)
        n2.append(domain.lines2d[i[2]].n)
        p0.append(domain.lines2d[i[0]].points[0].p)   
        p0_e.append(domain.lines2d[i[0]].points[1].p)   
    
    n0 = np.array(n0)
    n1 = np.array(n1)
    n2 = np.array(n2)
    p0 = np.array(p0)
    p0_e = np.array(p0_e)
    
    a1 = np.dot(n0, np.concatenate((R0,np.zeros((3,6))), axis=1))
    a2 = np.dot(n1, np.concatenate((R1,np.eye(3), np.zeros((3,3))), axis=1))
    a3 = np.dot(n2, np.concatenate((R2,np.zeros((3,3)), np.eye(3)), axis=1))
    
    b = np.dot(p0, R0)
    b = b
    b1 = np.concatenate((np.zeros((len(p0),1)), (-b[:,2]).reshape((len(p0),1)), (b[:,1]).reshape((len(p0),1)), np.zeros((len(p0),6))), axis=1)
    b2 = np.concatenate(((b[:,2]).reshape((len(p0),1)), np.zeros((len(p0),1)), (-b[:,0]).reshape((len(p0),1)), np.zeros((len(p0),6))), axis=1)
    b3 = np.concatenate(((-b[:,1]).reshape((len(p0),1)), (b[:,0]).reshape((len(p0),1)), np.zeros((len(p0),7))), axis=1)

    a4 = [] #will contain the equations from b1, b2, b3 LI to a1
    #Checking if a1 are LI to bi. If so, will be add to the matrix A
    for i, v1 in enumerate(a1):
        V2 = [b1[i], b2[i], b3[i]]
        for v2 in V2:
            if is_lin_ind(v1,v2)=="LI":
                a4.append(v2)
                break
    a4 = np.array(a4)
      
    
    Ai = [] #Initial matrix A without adding full elements to each equation
    
    #Here the matrix contain 9 columns withoug respecting diferent X variables
    for i in range(len(p0)):
        Ai.append(a1[i])
        Ai.append(a2[i])
        Ai.append(a3[i])
        Ai.append(a4[i]) #if want to check LI among 4 equations
        #Ai.append(b1[i]) #if just taking first equation of cross product as the LI
        #Ai.append(b2[i]) #if just taking first equation of cross product as the LI
        
    Ai = np.array(Ai)
    A = np.zeros((4*len(p0), 6+len(p0)*3))
    
    
    for i in range(len(p0)):
        A[i*4:(i+1)*4,i*3:(i+1)*3] = Ai[i*4:(i+1)*4,:3]
        A[i*4:(i+1)*4,-6:] = Ai[i*4:(i+1)*4,-6:]
    
    Ai_rezaie = np.vstack((a1,a2,a3,b1,b2,b3))
    A_rezaie = np.zeros((6*len(p0), 6+len(p0)*3))
    
    for i in range(6):
        for j in range(len(p0)):
            A_rezaie[i*5+j,j*3:(j+1)*3] = Ai_rezaie[i*5+j,:3]
            A_rezaie[i*5+j,-6:] = Ai_rezaie[i*5+j,-6:]
            #print(A_rezaie[i*5+j])
    
    
    
    AAi = [] #Initial matrix A without adding full elements to each equation
    #Here the matrix contain 9 columns withoug respecting diferent X variables
    for i in range(len(p0)):
        AAi.append(a1[i])
        AAi.append(a2[i])
        AAi.append(a3[i])
        AAi.append(b1[i])
        AAi.append(b2[i])
        AAi.append(b3[i])
        
    AAi = np.array(AAi)
    AA = np.zeros((6*len(p0), 6+len(p0)*3))
    
    for i in range(len(p0)):
        AA[i*6:(i+1)*6,i*3:(i+1)*3] = AAi[i*6:(i+1)*6,:3]
        AA[i*6:(i+1)*6,-6:] = AAi[i*6:(i+1)*6,-6:]
    
    #return A, A_rezaie, AA, n0,n1,n2,p0
    #return AA, n0,n1,n2,p0, p0_e
    return AA

def nullspace(A, atol=1e-13, rtol=0):
    """
    Description
    -------
    Compute an approximate basis for the nullspace of A.
    The algorithm used by this function is based on the singular value
    decomposition of `A`.
    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.
    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.
    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def find_view_rotation(domain, lines_v):
    """
    Description
    --------
    Finds the view rotation in relation with vanishing points directions
    represented by 2 parallel lines and one perpendicular to them in 3D
    Follows eq. 4 paper

    Parameters
    ----------
    domain : Domain
        Domain object with information of structure and motion.
    lines_v : list
        List with 3 lines ids

    Returns
    -------
    numpy.ndarray
        View rotation in relation with vanishing point directions.

    """
    
    n_01 = domain.lines2d[lines_v[0]].n
    n_02 = domain.lines2d[lines_v[1]].n
    n_03 = domain.lines2d[lines_v[2]].n
    d_01_a = np.cross(n_01,n_02)
    d_01_b = -np.cross(n_01,n_02)
    d_02_a = np.cross(d_01_a,n_03)
    d_02_b = np.cross(d_01_b,n_03)
    
    d_01 = [d_01_a, d_01_b]
    d_02 = [d_02_a, d_02_b]
    
    #Ri_kl has got 4 options (a,b,c,d)
    Rj = []
    for i in range(len(d_01)):
        for j in range(len(d_02)):
            r01 = (d_01[i]/np.linalg.norm(d_01[i])).reshape((3,1))
            r02 = (d_02[j]/np.linalg.norm(d_02[j])).reshape((3,1))
            r03 = ((np.cross(d_01[i],d_02[j]))/np.linalg.norm(np.cross(d_01[i],d_02[j]))).reshape((3,1))
            r0 = np.hstack((r01,r02,r03))
            Rj.append(r0)
    
    return np.array(Rj)

def find_relative_rotation(Rj_i, Rj_i1):
    """
    Description
    -------
    Following Eq. (5) it is found the relative rotation

    Parameters
    ----------
    Rj_i : numpy.ndarray
        view i rotation related to vanishing directions.
    Rj_i1 : numpy.ndarray
        view i+1 rotation related to vanishing directions.

    Returns
    -------
    TYPE
        Relative rotations views i adn i+1.

    """
    #Follows eq. 5 paper
    R = [] #Relative rotation matrix R_kl

    for i in range(len(Rj_i)):
        for j in range(len(Rj_i1)):
            R.append(np.dot(Rj_i1[j],Rj_i[i].T))
    
    R = np.array(R)
    #print(len(R))
    R = np.unique(R, axis=0) #4 different possibilites instead of 16
    #print(len(R))
    
    #Selecting the matrix with the minimum rotation
    #R = R[np.argmin([np.linalg.norm(rotation_matrix_as_euler_angles(i)) for i in R])]
    R = R[np.argmin([np.linalg.norm(rotation_matrix_as_euler_angles(i)) for i in R])]
    
    return np.array(R)


def find_correspondences_3v(domain, v0, v1, v2, n_corr=5):
    """
    Description
    -------
    Find correspondances in 3 views

    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    v0 : int
        DESCRIPTION.
    v1 : int
        DESCRIPTION.
    v2 : int
        DESCRIPTION.
    n_corr : int or str, optional
        if "full" it will give all correspondances. if int, it will give 
        aleatory n_corr. The default is 5.

    Returns
    -------
    numpy.ndarray
        Returns n_corr or "full" correspondences in 3 views.

    """

    #Finds correspondences among three views (v0,v1,v2) and return n_corr aleatory correspondances
    corr_3v = []

    for l_01 in domain.line_match[str(v0)+str(v1)][1]:
        for l_12 in domain.line_match[str(v1)+str(v2)][1]:
            if l_01[1]==l_12[0]: #if line of view matched in 01 also is matched in 12, append 3view correspondancse
                corr_3v.append(l_01+[l_12[1]])
    corr_3v = np.array(corr_3v)
    ind_n_corr = np.array(range(len(corr_3v)))
    np.random.shuffle(ind_n_corr)
    
    if n_corr == "full":
        return np.array(corr_3v)
    else:    
        return np.array(corr_3v[:n_corr])

def find_correspondences_3v_mod(domain, v0, v1, v2, n_corr=5):
    """
    Description
    -------
    Find correspondances in 3 views, Modified version    

    Parameters
    ----------
    domain : TYPE
        DESCRIPTION.
    v0 : TYPE
        DESCRIPTION.
    v1 : TYPE
        DESCRIPTION.
    v2 : TYPE
        DESCRIPTION.
    n_corr : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Finds correspondences among three views (v0,v1,v2) and return n_corr aleatory correspondances
    corr_3v = []

    for l_01 in domain.line_match[str(v0)+str(v1)][1]:
        for l_12 in domain.line_match[str(v1+str(v2))][1]:
            if l_01[1] == l_12[0]:  # if line of view matched in 01 also is matched in 12, append 3view correspondancse
                corr_3v.append(l_01 + [l_12[1]])
    corr_3v = np.array(corr_3v)
    number_of_rows = corr_3v.shape[0]
    random_indices = np.random.choice(number_of_rows, size=n_corr, replace=False)
    
    return corr_3v[random_indices,:]

def plot_rotations(R0,R1,R2,x,x_amir):
    """
    Description
    -------
    Given 3 rotation matrices and null space containing translations,
    Plot how vector [0,0,1] is transformed

    Parameters
    ----------
    R0 : numpy.ndarray
        Rotation view 0.
    R1 : numpy.ndarray
        Rotation view 1.
    R2 : numpy.ndarray
        Rotation view 2.
    x : numpy.ndarray
        null space of A.        
    x_amir : numpy.ndarray
        null space of A (AR).

    Returns
    -------
    None.

    """
    #Ploting rotations

    
    xc = np.array([[0],[0],[1]])
    xr0 = R0@xc
    xr1 = R1@xc
    xr1 = xr1/np.linalg.norm(xr1)
    xr2 = R2@xc
    xr2 = xr2/np.linalg.norm(xr2)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot([0, xc[0][0]], [0,xc[1][0]],zs=[0,xc[2][0]], c='red')
    ax.plot([0, xr0[0][0]], [0,xr0[1][0]],zs=[0,xr0[2][0]], c='blue')
    ax.plot([0, xr1[0][0]], [0,xr1[1][0]],zs=[0,xr1[2][0]], c='green')
    ax.plot([0, xr2[0][0]], [0,xr2[1][0]],zs=[0,xr2[2][0]], c='yellow')
    
    xc = np.array([[0],[0],[1]])
    xr0 = R0@xc
    xr1 = R1@xc
    xr1 = xr1/np.linalg.norm(xr1)
    xr2 = R2@xc
    xr2 = xr2/np.linalg.norm(xr2)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot([0, xc[0][0]], [0,xc[1][0]],zs=[0,xc[2][0]], c='red')
    ax.plot([0, xr0[0][0]], [0,xr0[1][0]],zs=[0,xr0[2][0]], c='blue')
    ax.plot([x[-6][0], xr1[0][0]]+x[-6][0], [x[-5][0],xr1[1][0]+x[-5][0]],zs=[x[-4][0],xr1[2][0]+x[-4][0]], c='green')
    ax.plot([x[-3][0], xr2[0][0]]+x[-3][0], [x[-2][0],xr2[1][0]+x[-2][0]],zs=[x[-1][0],xr2[2][0]+x[-1][0]], c='yellow')
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot([0, xc[0][0]], [0,xc[1][0]],zs=[0,xc[2][0]], c='red')
    ax.plot([0, xr0[0][0]], [0,xr0[1][0]],zs=[0,xr0[2][0]], c='blue')
    ax.plot([x_amir[-6][0], xr1[0][0]]+x_amir[-6][0], [x_amir[-5][0],xr1[1][0]+x_amir[-5][0]],zs=[x_amir[-4][0],xr1[2][0]+x_amir[-4][0]], c='green')
    ax.plot([x_amir[-3][0], xr2[0][0]]+x_amir[-3][0], [x_amir[-2][0],xr2[1][0]+x_amir[-2][0]],zs=[x_amir[-1][0],xr2[2][0]+x_amir[-1][0]], c='yellow')

def plot_3D_lines_from_x(R0, R1, R2, n0, n1, n2, p0, p0_e):
    """
    Description
    -------
    Using null space x from matrix A, it is plot the 3D lines. It is 
    to verify if equation proposed in the paper Ax=0 gives appropriate values
    for X
    

    Parameters
    ----------
    R0 : numpy.ndarray
        Rotation camera 0.
    R1 : numpy.ndarray
        Rotation camera 1.
    R2 : numpy.ndarray
        Rotation camera 2.
    n0 : numpy.ndarray
        Normale vector to interpretation plane 0.
    n1 : numpy.ndarray
        Normale vector to interpretation plane 1.
    n2 : numpy.ndarray
        Normale vector to interpretation plane 2.
    p0 : numpy.ndarray
        Spherical representation of initial endpoint in line 0.
    p0_e : numpy.ndarray
        Spherical representation of final endpoint in line 0.

    Returns
    -------
    None.

    """
    ##PLOTING 3D LINES###
    A_start = create_matrix_A(R0, R1, R2, n0[0, :], n0[1, :], n0[2, :], n0[3, :], n0[4, :],
                                          n1[0, :], n1[1, :], n1[2, :], n1[3, :], n1[4, :],
                                          n2[0, :], n2[1, :], n2[2, :], n2[3, :], n2[4, :],
                                          p0[0, :], p0[1, :], p0[2, :], p0[3, :], p0[4, :])
    x_start = scipy.linalg.null_space(A_start)
    
    A_end = create_matrix_A(R0, R1, R2, n0[0, :], n0[1, :], n0[2, :], n0[3, :], n0[4, :],
                                          n1[0, :], n1[1, :], n1[2, :], n1[3, :], n1[4, :],
                                          n2[0, :], n2[1, :], n2[2, :], n2[3, :], n2[4, :],
                                          p0_e[0, :], p0_e[1, :], p0_e[2, :], p0_e[3, :], p0_e[4, :])
    x_end = scipy.linalg.null_space(A_end)
    
    xs_start = x_start[0::3][0:5]
    ys_start = x_start[1::3][0:5]
    zs_start = x_start[2::3][0:5]
    
    xs_end = x_end[0::3][0:5]
    ys_end = x_end[1::3][0:5]
    zs_end = x_end[2::3][0:5]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for line_ind in [0, 1, 2, 3, 4]:
        line_start = np.array([xs_start[line_ind][0], ys_start[line_ind][0], zs_start[line_ind][0]])
        line_end = np.array([xs_end[line_ind][0], ys_end[line_ind][0], zs_end[line_ind][0]])
        ax.plot([line_start[0], line_end[0]],
                [line_start[1], line_end[1]],
                [line_start[2], line_end[2]], c='red')

def compute_epipole(F):
    """
    Description
    -------
    Computes the (right) epipole from a 
    fundamental matrix F. 
        (Use with F.T for left epipole.)

    Parameters
    ----------
    F : numpy.ndarray
        Fundamental matrix.

    Returns
    -------
    umpy.ndarray
        epipole.

    """
    
    # return null space of F (Fx=0)
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]

def conv_pt_homogeneus(x_i):
    """
    Description
    -------
    Conver 2d or 3d point to homogeneus coordinates    
    
    Parameters
    ----------
    x_i : numpy.ndarray
        2D or 3D point coordinates.
    Returns
    -------
    numpy.ndarray
        Point in homogeneus coordinates.
    """
    x_i = np.array(x_i)
    x_i = np.concatenate((x_i, np.array([1])))
    if len(x_i)==3:    
        return x_i.reshape((3,1))
    elif len(x_i)==4:
        return x_i.reshape((4,1))

def plot_epipolar_line(im_i, im_i1, F, x_i, epipole=None, show_epipole=False):
    """
    Description
    -------
    Plot the epipole and epipolar line F*x_i=0
    in an image. F is the fundamental matrix 
    and x_i a point in the other image.
        

    Parameters
    ----------
    im_i : numpy.ndarray
        Image i.
    im_i1 : numpy.ndarray
        Image i+1.
    F : numpy.ndarray
        Fundamental matrix between images i, i+1.
    x_i : numpy.ndarray
        Point in view i to be projected in image i+1 as epipolar line.
    epipole : TYPE, optional
        epipole coordinates. The default is None.
    show_epipole : TYPE, optional
        If true print epipole in image i+1. The default is False.

    Returns
    -------
    None.

    """

    x_i = conv_pt_homogeneus(x_i)    
    
    sh0 = im_i.shape
    sh1 = im_i1.shape
    shift_x = sh1[1]
    im = np.zeros((max(sh0[0],sh1[0]), sh0[1]+sh1[1], 3))
    im[0:sh0[0], 0:sh0[1], :] = im_i
    im[0:sh1[0], sh0[1]:, :] = im_i1
    im = im.astype('uint8')
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    m, n = im_i1.shape[:2]
    #n,m = im.shape[:2]
    line = np.dot(F, x_i)
    
    # epipolar line parameter and values
    t = np.linspace(0,n,100).reshape((-1,1))
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt>=0) & (lt<m) 
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.plot(t[ndx]+shift_x,lt[ndx],linewidth=2)
    #plt.plot(t[ndx], lt[ndx],linewidth=2)
    plt.scatter(x_i[0], x_i[1], s=10, c='red')
    
    
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')

def F_from_cameras(P0, P1):
    """
    Description
    -------
    #see https://sourishghosh.com/2016/fundamental-matrix-from-camera-matrices/    
    
    Computes fundamental matrix from two views when given two camera objects

    Parameters
    ----------
    P0 : Camera
        Camera object view 0.
    P1 : Camera
        Camera object view 1.

    Returns
    -------
    numpy.ndarray
        Fundamental camera matrix views 0-1.
    
    """
    
    
    C0 = conv_pt_homogeneus(P0.c) #Camera center view 0
    e1 = P1.P @ C0 #epipole in view 1
    P0_pinv = np.linalg.pinv(P0.P)
    e1x = np.array([[0,-e1[2][0], e1[1][0]],
                    [e1[2][0], 0, -e1[0][0]],
                    [-e1[1][0], e1[0][0], 0]])#cross product matrix
    
    
    F = e1x @ P1.P @ P0_pinv

    return F/F[2,2] #element 2,2 needs to be 1

def plot_epipolar_lines_plus_line(im_i, im_i1, F, l_i, l_i1):
    """
    Description
    -------
    Plot a pair of line corresponcences. Plot also the two epipolar lines correspondant
    to the first view line endpoints over second view. This helps to check
    how accurate are second view end points correspondances and posibly correct them
    
    Parameters
    ----------
    im_i : numpy.ndarray
        Image array view i.
    im_i1 : numpy.ndarray
        Image array view i+1.
    F : numpy.ndarray
        Fundamental matrix views i, i+1.
    l_i : Line
        Correspondence line from view i.
    l_i1 : Line
        Correspondence line from view i+1.
    Returns
    -------
    None.
    """
    
    x_i_0 = conv_pt_homogeneus(l_i.points[0].coord)
    x_i_1 = conv_pt_homogeneus(l_i.points[1].coord)
    
    x_i1_0 = conv_pt_homogeneus(l_i1.points[0].coord)
    x_i1_1 = conv_pt_homogeneus(l_i1.points[1].coord)
    
    sh0 = im_i.shape
    sh1 = im_i1.shape
    shift_x = sh1[1]
    im = np.zeros((max(sh0[0],sh1[0]), sh0[1]+sh1[1], 3))
    im[0:sh0[0], 0:sh0[1], :] = im_i
    im[0:sh1[0], sh0[1]:, :] = im_i1
    im = im.astype('uint8')
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    m, n = im_i1.shape[:2]
    line0 = np.dot(F, x_i_0)
    line1 = np.dot(F, x_i_1)
    
    # epipolar line parameter and values
    t = np.linspace(0,n,100).reshape((-1,1))
    lt0 = np.array([(line0[2]+line0[0]*tt)/(-line0[1]) for tt in t])
    lt1 = np.array([(line1[2]+line1[0]*tt)/(-line1[1]) for tt in t])
    
    # take only line points inside the image
    ndx0 = (lt0>=0) & (lt0<m) 
    ndx1 = (lt1>=0) & (lt1<m) 
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.plot(t[ndx0]+shift_x,lt0[ndx0],linewidth=2, c='blue')
    plt.plot(t[ndx1]+shift_x,lt1[ndx1],linewidth=2, c='blue')
    #plt.plot(t[ndx0],lt0[ndx0],linewidth=2, c='blue')
    #plt.plot(t[ndx1],lt1[ndx1],linewidth=2, c='blue')
    
    plt.plot([x_i_0[0], x_i_1[0]], [x_i_0[1], x_i_1[1]],linewidth=2, c='green')
    plt.plot([x_i1_0[0]+shift_x, x_i1_1[0]+shift_x], [x_i1_0[1], x_i1_1[1]],linewidth=2, c='green')
    
    plt.scatter(x_i_0[0], x_i_0[1], s=10, c='red')
    plt.scatter(x_i_1[0], x_i_1[1], s=10, c='red')

def F_from_KRt(K, rR, rt):
    """
    Description
    -------
    see https://sourishghosh.com/2016/fundamental-matrix-from-camera-matrices/
    
    Computes two views F from relative translation, relative rotation 
    and intrinsic matrix
    
    Parameters
    ----------
    K : TYPE
        Intrinsics matrix.
    rR : TYPE
        Relative rotation between v0 and v1.
    rt : TYPE
        Relative translation between v0 and v1.
    Returns
    -------
    F : TYPE
        Fundamental Matrix.
    """
    
    
    
    
    term = K @ rR.T @ rt.reshape((3,1))
    termx = np.array([[0,-term[2][0], term[1][0]],
                [term[2][0], 0, -term[0][0]],
                [-term[1][0], term[0][0], 0]])#cross product matrix
    
    F = np.linalg.inv(K).T @ rR @ K.T @ termx
    
    return F/F[2,2] #element 2,2 needs to be 1

def F_from_KRt2(K, rR, rt):
    """
    Description
    -------
    see https://sourishghosh.com/2016/fundamental-matrix-from-camera-matrices/
    
    Computes two views F from relative translation, relative rotation 
    and intrinsic matrix
    
    Parameters
    ----------
    K : TYPE
        Intrinsics matrix.
    rR : TYPE
        Relative rotation between v0 and v1.
    rt : TYPE
        Relative translation between v0 and v1.
    Returns
    -------
    F : TYPE
        Fundamental Matrix.
    """
    
    rt = rt.reshape((3,1))
    
    rt_x = np.array([[0,-rt[2][0], rt[1][0]],
                [rt[2][0], 0, -rt[0][0]],
                [-rt[1][0], rt[0][0], 0]])#cross product matrix
    
    F = np.linalg.inv(K).T @ rt_x @ rR @ np.linalg.inv(K)
    
    return F/F[2,2] #element 2,2 needs to be 1


def test_triangulation_P_from_sift(domain, K):
    """
    Description
    ------
    Various test to verify cameras computed with equations from paper
    
    This uses P matrices from sift and triangulates 4 scenaries
    #1: matches views 01
    #2: matches views 12
    #3: 1+2 in the same plot
    #4: matches views 02
    #Also computes F from implemented functions and compares with F from sift

    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    K : numpy.ndarray
        Matrix with intrinsic parameters of the camera.

    Returns
    -------
    None.

    """
        
    #Creating Projective Camera Matrices
    P0 = Camera(domain.P_rel[0][1])
    P0.factor()
    P0.center()
    P1 = Camera(domain.P_rel[0][2])
    P1.factor()
    P1.center()
    #for P2 it is needed to find R2 and t2 global as they are relative
    R12 = (np.linalg.inv(K) @ domain.P_rel[1][2])[:,:3]
    t12 = (np.linalg.inv(K) @ domain.P_rel[1][2])[:,3]
    R2 = R12 @ P1.R
    t2 = t12 + (R12 @ P1.t)
    #t2 = t12 + P1.t
    t2 = t2.reshape((3,1))
    P2 = Camera(K @ np.hstack((R2,t2)))
    P2.factor()
    P2.center()
    
    #Triangulate points views 0-1
    #Gathering kps for each view
    kp0 = domain.kps_match[0][2].T
    kp1 = domain.kps_match[0][3].T
    #Triangulating points
    X01 = cv2.triangulatePoints(P0.P, P1.P, kp0, kp1)
    #Dividing by homogeneus coordinate
    X01 = X01[:]/X01[3]
    
    #Box plots to see outilers (plot just 2 values among 10 adn 90% quartile)
    q0 = np.quantile(X01[0],(0,0.1,.5,.9,1))
    q1 = np.quantile(X01[1],(0,0.1,.5,.9,1))
    q2 = np.quantile(X01[2],(0,0.1,.5,.9,1))
    
    #Ploting points without outliers
    fig = plt.figure()
    #inliers_ind = np.where((np.abs(X[0]) < 0.1)  & (np.abs(X[1]) < 0.1) & (np.abs(X[2]) < 0.1))[0]
    inliers_ind = np.where((q0[1] < X01[0]) & (X01[0] < q0[3])  & (q1[1] < X01[1]) & (X01[1] < q1[3]) & (q2[1] < X01[2]) & (X01[2] < q2[3]))[0]
    ax = fig.gca(projection='3d')
    ax.scatter(X01[0][inliers_ind],X01[1][inliers_ind], X01[2][inliers_ind], s=0.1, c='r')
    plt.axis('on')
    
    
    ##Triangulate points views 1-2
    #Gathering kps for each view
    kp1 = domain.kps_match[1][2].T
    kp2 = domain.kps_match[1][3].T
    #Triangulating points
    X12 = cv2.triangulatePoints(P1.P, P2.P, kp1, kp2)
    #X = cv2.triangulatePoints(domain.P_rel[1][1], domain.P_rel[1][2], kp1, kp2)
    #Dividing by homogeneus coordinate
    X12 = X12[:]/X12[3]
    
    #Box plots to see outilers (plot just 2 values among 10 adn 90% quartile)
    q0 = np.quantile(X12[0],(0,0.1,.5,.9,1))
    q1 = np.quantile(X12[1],(0,0.1,.5,.9,1))
    q2 = np.quantile(X12[2],(0,0.1,.5,.9,1))
    
    #Ploting points without outliers
    fig = plt.figure()
    #inliers_ind = np.where((np.abs(X[0]) < 0.1)  & (np.abs(X[1]) < 0.1) & (np.abs(X[2]) < 0.1))[0]
    inliers_ind = np.where((q0[1] < X12[0]) & (X12[0] < q0[3])  & (q1[1] < X12[1]) & (X12[1] < q1[3]) & (q2[1] < X12[2]) & (X12[2] < q2[3]))[0]
    ax = fig.gca(projection='3d')
    ax.scatter(X12[0][inliers_ind],X12[1][inliers_ind], X12[2][inliers_ind], s=0.1, c='r')
    plt.axis('on')
    
    
    #Ploting in the same figure points X01 and X12
    X012 = np.concatenate((X01, X12), axis=1)
    #Box plots to see outilers (plot just 2 values among 10 adn 90% quartile)
    q0 = np.quantile(X012[0],(0,0.05,.5,.95,1))
    q1 = np.quantile(X012[1],(0,0.05,.5,.95,1))
    q2 = np.quantile(X012[2],(0,0.05,.5,.95,1))
    
    #Ploting points without outliers
    fig = plt.figure()
    #inliers_ind = np.where((np.abs(X[0]) < 0.1)  & (np.abs(X[1]) < 0.1) & (np.abs(X[2]) < 0.1))[0]
    inliers_ind = np.where((q0[1] < X012[0]) & (X012[0] < q0[3])  & (q1[1] < X012[1]) & (X012[1] < q1[3]) & (q2[1] < X012[2]) & (X012[2] < q2[3]))[0]
    ax = fig.gca(projection='3d')
    ax.scatter(X012[0][inliers_ind],X012[1][inliers_ind], X012[2][inliers_ind], s=0.1, c='r')
    plt.axis('on')
    
    
    ##Triangulate points views 0-2#################
    #Gathering kps for each view
    kp0 = np.array(domain.views[0].kps)
    kp2 = np.array(domain.views[2].kps)
    
    #Reading descriptors
    desc0 = np.array(domain.views[0].desc_kps)
    desc2 = np.array(domain.views[2].desc_kps)
    
    #Normalizing descriptors
    desc0 = desc0/np.linalg.norm(desc0)
    desc2 = desc2/np.linalg.norm(desc2)
    
    #print(len(kp1))
    
    #Matching using sift criteria            
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc0, desc2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    matchesMask = [[0,0] for i in range(len(matches))]
    for i, (m,n) in enumerate(matches):
        #if m.distance < 0.7*n.distance:
        if m.distance < 0.7*n.distance:
            good.append(m)
            matchesMask[i]=[1,0]
    if len(good)>10:
        src_pts = np.float32([ kp0[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    kp0 = src_pts.reshape((len(src_pts),2)).T
    kp2 = dst_pts.reshape((len(dst_pts),2)).T
    
    #Triangulating points
    X02 = cv2.triangulatePoints(P0.P, P2.P, kp0, kp2)
    #X = cv2.triangulatePoints(domain.P_rel[1][1], domain.P_rel[1][2], kp1, kp2)
    #Dividing by homogeneus coordinate
    X02 = X02[:]/X02[3]
    
    #Box plots to see outilers (plot just 2 values among 10 adn 90% quartile)
    q0 = np.quantile(X02[0],(0,0.1,.5,.9,1))
    q1 = np.quantile(X02[1],(0,0.1,.5,.9,1))
    q2 = np.quantile(X02[2],(0,0.1,.5,.9,1))
    
    #Ploting points without outliers
    fig = plt.figure()
    #inliers_ind = np.where((np.abs(X[0]) < 0.1)  & (np.abs(X[1]) < 0.1) & (np.abs(X[2]) < 0.1))[0]
    inliers_ind = np.where((q0[1] < X02[0]) & (X02[0] < q0[3])  & (q1[1] < X02[1]) & (X02[1] < q1[3]) & (q2[1] < X02[2]) & (X02[2] < q2[3]))[0]
    ax = fig.gca(projection='3d')
    ax.scatter(X02[0][inliers_ind],X02[1][inliers_ind], X02[2][inliers_ind], s=0.1, c='r')
    plt.axis('on')


    #Checking if F functions from P work (They dont give same results)
    
    print("F01 from sift" , domain.F_matrices[0][1])
    print("F01 from f1" , F_from_cameras(P0,P1))
    print("F01 from f2" , F_from_KRt(K, P1.R, P1.t))
    print("F01 from f3" , F_from_KRt2(K, P1.R, P1.t))
    
    print("F12 from sift" , domain.F_matrices[1][1])
    print("F12 from f1" , F_from_cameras(P1,P2))
    print("F12 from f2" , F_from_KRt(K, P2.R, P2.t))
    print("F12 from f3" , F_from_KRt2(K, P2.R, P2.t))


def line_parametrization(x_i1_0, x_i1_1):
    """
    Description
    ------
    Parametrize a line 2D with 3 parameters A,B,C

    Parameters
    ----------
    x_i1_0 : numpy.ndarray
        Coordinates initial point line.
    x_i1_1 : numpy.ndarray
        Coordinates final point line.

    Returns
    -------
    A : float
        line parameter.
    B : float
        line parameter.
    C : float
        line parameter.

    """
    #creates a line representations from 2 endpoints
    A = (x_i1_0[1] - x_i1_1[1])
    B = (x_i1_1[0] - x_i1_0[0])
    C = (x_i1_0[0]*x_i1_1[1] - x_i1_1[0]*x_i1_0[1])
    return (A,B,C)

def intersection(L1, L2):
    """
    Description
    -------
    #computes intersection between lines in their parametrizated form
    
    Parameters
    ----------
    L1 : tuple
        parametrized line 2D 1.
    L2 : tuple
        parametrized line 2D 1.

    Returns
    -------
    float, float or bool
        Line intersection if found. If not False

    """
    
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = -L1[2] * L2[1] + L1[1] * L2[2]
    Dy = -L1[0] * L2[2] + L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False




def find_endpts_correspondences_corrected(domain, view, K, F=None, plot=False, num_plot=5):
    """
    Given the initial view, it computes the end point correspondances of 
    line segments correcting the segments in the viewi1 using their 
    intersection with epipolar lines
    This is the version 0, I thought I could use the same function for ransac
    but maybe is better to separate them

    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    view : int
        view i id.
    K : numpy.ndarray
        Intrinsic parameters matrix.
    F : numpy.ndarray, optional
        Fundamental matrix views i, i+1. The default is None.
    plot : bool, optional
        If true, plot epipolar line for points. The default is False.
    num_plot : int, optional
        number of plots to be shown. The default is 5.

    Returns
    -------
    endpts_v0 : numpy.ndarray
        Endpoints correspondences view i.
    endpts_v1_corr : numpy.ndarray
        Endpoints correspondences corrected with epipolar lines view i+1.
    l_matches_inliers : numpy.ndarray
        Inlier end point correspondences.

    """

    
    #l_matches = domain.line_match[view]
    l_matches = domain.line_match[str(view)+str(view+1)]
    im_i = domain.views[view].image # Image i
    im_i1 = domain.views[view+1].image # Image i+1
    if F is None:
        #F = domain.F_matrices[view][1] #fundamental matrix from sift keypoints 
        #print(domain.P_rel[view][3])
        F = np.linalg.inv(K).T @ domain.P_rel[view][3] @ np.linalg.inv(K) #fundamental matrix from sift keypoints
        F/=F[2,2]
        #print(F)
    
    #Creating image if ploting
    sh0 = im_i.shape
    sh1 = im_i1.shape
    shift_x = sh1[1]
    
    im = np.zeros((max(sh0[0],sh1[0]), sh0[1]+sh1[1], 3))
    im[0:sh0[0], 0:sh0[1], :] = im_i
    im[0:sh1[0], sh0[1]:, :] = im_i1
    im = im.astype('uint8')    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   
    m, n = im_i1.shape[:2]
    
    #l_i = domain.lines2d[549]
    #l_i1 = domain.lines2d[1689]
    endpts_v0 = [] #end points correspondences v0
    endpts_v1 = [] #end points correspondences v1
    endpts_v1_corr = [] #end points correspondences corrected v1
    pr = 0
    num_inl = 0 #Number of inliers to give score 1
    l_matches_inliers = [] #List with the inlier matches
    
    
    
    for lm in l_matches[1]:
        #if pr%100==0:
            #print("progress correcting endpoints: ", np.round(100*pr/len(l_matches[1]), 2), "%")
        
        #print("Correcting endpoints and computing inliers based on epipolar lines---------------------")
        
        pr+=1
        l_i = domain.lines2d[lm[0]]
        l_i1 = domain.lines2d[lm[1]]
        
        #Making endpoints homogeneus
        x_i_0 = conv_pt_homogeneus(l_i.points[0].coord)
        x_i_1 = conv_pt_homogeneus(l_i.points[1].coord)
        x_i1_0 = conv_pt_homogeneus(l_i1.points[0].coord)
        x_i1_1 = conv_pt_homogeneus(l_i1.points[1].coord)
        
        #Finding line representation for segment in view i1
        line_i1 = line_parametrization(x_i1_0, x_i1_1)        

        line0 = np.dot(F, x_i_0)
        line1 = np.dot(F, x_i_1)
        
        #Finding interseccions between line segment and epipolar lines
        xi0,yi0 = intersection(line_i1, line0)
        xi1,yi1 = intersection(line_i1, line1)
        
        #Thresold to the distance between endpoints (x0,x1) and corrected ones (xp,xq). 
        #1) Distance of at least one end point has to be is lower than 10% of the min(||xp-xq||, ||x0-x1||)
        #2) the second endpoint has to be closer to the other intersection point. (not too clear)
        len_line = np.linalg.norm(np.array([xi1[0],yi1[0]])-np.array([xi0[0],yi0[0]]))
        len_line_corr = np.linalg.norm(np.array(l_i1.points[1].coord)-np.array(l_i1.points[0].coord))
        thr = 0.01*min(len_line,len_line_corr)
        
        #Checking if match meets the threshold (disti_j: distance endpoint i to inters j)
        dist1_1 = np.linalg.norm(np.array(l_i1.points[0].coord)-np.array([xi0[0],yi0[0]]))
        dist2_2 = np.linalg.norm(np.array(l_i1.points[1].coord)-np.array([xi1[0],yi1[0]]))
        
        dist1_2 = np.linalg.norm(np.array(l_i1.points[0].coord)-np.array([xi1[0],yi1[0]]))
        dist2_1 = np.linalg.norm(np.array(l_i1.points[1].coord)-np.array([xi0[0],yi0[0]]))
        
        #if dist1_1<thr or dist2_2 < thr: #Criteria 1)
        
            #if dist1_1<dist1_2 and dist2_2<dist2_1:
        
        l_matches_inliers.append(lm)
        num_inl+=1
        
        endpts_v0.append(l_i.points[0].coord)
        endpts_v0.append(l_i.points[1].coord)
        endpts_v1.append(l_i1.points[0].coord)
        endpts_v1.append(l_i1.points[1].coord)
        endpts_v1_corr.append([xi0,yi0])
        endpts_v1_corr.append([xi1,yi1])                
    
        if plot and num_inl < num_plot:
            
            #PLOTING: USED TO VERIFY IF INTERSECTIONS ARE OK
            # epipolar line parameter and values
            t = np.linspace(0,n,100).reshape((-1,1))
            lt0 = np.array([(line0[2]+line0[0]*tt)/(-line0[1]) for tt in t])
            lt1 = np.array([(line1[2]+line1[0]*tt)/(-line1[1]) for tt in t])
            lt_i1 = np.array([(line_i1[2]+line_i1[0]*tt)/(-line_i1[1]) for tt in t])
            
            # take only line points inside the image
            ndx0 = (lt0>=0) & (lt0<m) 
            ndx1 = (lt1>=0) & (lt1<m) 
            ndx_i1 = (lt_i1>=0) & (lt_i1<m) 
            plt.figure()
            plt.imshow(im, cmap='gray')
            plt.plot(t[ndx0]+shift_x,lt0[ndx0],linewidth=2, c='blue')
            plt.plot(t[ndx1]+shift_x,lt1[ndx1],linewidth=2, c='blue')
            
            #plt.plot(t[ndx0],lt0[ndx0],linewidth=2, c='blue')
            #plt.plot(t[ndx1],lt1[ndx1],linewidth=2, c='blue')
            
            plt.plot(t[ndx_i1]+shift_x,lt_i1[ndx_i1],linewidth=2, c='yellow')
            #plt.plot(t[ndx0],lt0[ndx0],linewidth=2, c='blue')
            #plt.plot(t[ndx1],lt1[ndx1],linewidth=2, c='blue')
            
            plt.plot([x_i_0[0], x_i_1[0]], [x_i_0[1], x_i_1[1]],linewidth=2, c='green')
            plt.plot([x_i1_0[0]+shift_x, x_i1_1[0]+shift_x], [x_i1_0[1], x_i1_1[1]],linewidth=2, c='green')
            
            plt.scatter(x_i_0[0], x_i_0[1], s=20, c='red')
            plt.scatter(x_i_1[0], x_i_1[1], s=20, c='red')
            plt.scatter(xi0+shift_x, yi0, s=20, c='red')
            plt.scatter(xi1+shift_x, yi1, s=20, c='red')
        
    endpts_v0 = np.array(endpts_v0).reshape((-1,2))
    endpts_v1 = np.array(endpts_v1).reshape((-1,2))
    endpts_v1_corr = np.array(endpts_v1_corr).reshape((-1,2))
        
    #Distances from inliers to give score 2
    distances = np.linalg.norm(endpts_v1[0::2]-endpts_v1_corr[0::2], axis=1) + np.linalg.norm(endpts_v1[0::2]-endpts_v1_corr[0::2], axis=1)
    #Filtering line matches
    #ind_filtered = distances<thr
    #l_matches_filtered = l_matches[1]
    
    return  endpts_v0, endpts_v1_corr, l_matches_inliers




def relativeCameraMotion(P0, P1):
    """
    Description
    -------
    Given Camera objects from two views, it returns their relative R and t

    Parameters
    ----------
    P0 : Camera
        Camera object view 0.
    P1 : Camera
        Camera object view 1.

    Returns
    -------
    rR : numpy.ndarray
        Relative rotation matrix.
    rt : numpy.ndarray
        Relative translation vector.

    """

    
    R0, t0 = P0.R, P0.t
    R1, t1 = P1.R, P1.t
    
    rR = R1 @ R0.T
    rt = t1 - rR @ t0

    return rR, rt


def read_line_logs(path, name):
    """
    Parameters
    ----------
    path : str
        The path to the .txt files generated by the lsd algorithm, which contains information about the detected lines.
    name : str
        The name of the file including the .txt extension.
    Returns
    -------
    A tuple of x1_arr, y1_arr, x2_arr, y2_arr, where
    x1_arr : array_like
        A numpy array containing all x1 coordinates of starting point of the detected lines.
    y1_arr : array_like
        A numpy array containing all y1 coordinates of starting point of the detected lines.
    x2_arr : array_like
        A numpy array containing all x2 coordinates of ending point of the detected lines.
    y1_arr : array_like
        A numpy array containing all y2 coordinates of ending point of the detected lines.
    """
    col_names = ['x1', 'y1', 'x2', 'y2', 'width', 'p', '-log_nfa']
    line_log_df = pd.read_csv(os.path.join(path, name),
                              delim_whitespace=True, header=None, names=col_names)
    x1_arr, y1_arr = line_log_df['x1'].to_numpy(), line_log_df['y1'].to_numpy()
    x2_arr, y2_arr = line_log_df['x2'].to_numpy(), line_log_df['y2'].to_numpy()
    return x1_arr, y1_arr, x2_arr, y2_arr


def get_three_lines(domain, view_id=0):
    """
    Description
    --------
    we find the ids of the first and second dominant parallel line ids in 
    the first view that are matched with the second view

    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    view_id : int, optional
        View id. The default is 0.

    Returns
    -------
    lines_vi : list
        List with lines ids vi.
    lines_vj : list
        List with lines ids vj.

    """
    
    allowed_par_line_ids_first_group = np.intersect1d(domain.views[view_id].parallel_line_ids[0],
                                                      np.array(domain.line_match[str(view_id)+str(view_id+1)][1])[:, 0])
    allowed_par_line_ids_second_group = np.intersect1d(domain.views[view_id].parallel_line_ids[1],
                                                       np.array(domain.line_match[str(view_id)+str(view_id+1)][1])[:, 0])
    lines_vi = []
    lines_vi.extend(np.random.choice(allowed_par_line_ids_first_group, size=2, replace=False))
    lines_vi.extend(np.random.choice(allowed_par_line_ids_second_group, size=1, replace=False))

    lines_vj = []
    matched_lines_vij = np.array(domain.line_match[str(view_id)+str(view_id+1)][1])
    for i in range(len(lines_vi)):
        row = np.argwhere(matched_lines_vij[:, 0] == lines_vi[i])[0, 0]
        lines_vj.append(matched_lines_vij[row, 1])
    return lines_vi, lines_vj

def rotration_as_quat(R):
    """
    Description
    -------
    Rotation representation as quaternion

    Parameters
    ----------
    R : numpy.ndarray
        Rotation matrix.

    Returns
    -------
    TYPE
        Vector with quaternion representation of Rotation matrix.

    """
    r = Rotation.from_matrix(R)
    return r.as_quat()

def generate_Rij_hyp(domain, view_id, n_iter):
    """
    Description
    -------
    Creates Relative rotations hypothesis for views i and j based in aleatory
    3 lines selected

    Parameters
    ----------
    domain : Domain
        Object from Domain class containing informations from structure and motion.
    view_id : int
        id assigned to view i.
    n_iter : int
        number of iterations in generation of Rij hypothesis. Number of hypothesis.

    Returns
    -------
    Rij_candid : numpy.ndarray
        Array containing relative rotation hypothesis.

    """
    Rij_candid = np.empty((n_iter, 3, 3))
    for iter in tqdm(range(n_iter)):
        lines_vi, lines_vj= get_three_lines(domain, view_id=view_id)
        #Calculating view rotations based on 3 lines
        Ri = find_view_rotation(domain, lines_vi)
        Rj = find_view_rotation(domain, lines_vj)
        # Compute relative rotations
        Rij = find_relative_rotation(Ri, Rj)
        Rij_candid[iter]=Rij
    return Rij_candid

# =============================================================================
# def find_R_optim(Rij_candid):
#     qij_candid = []
#     for Rij in Rij_candid:
#         qij_candid.append(rotration_as_quat(Rij))
# 
#     dist_Rn_R = np.empty((len(qij_candid), len(qij_candid)))
#     for i in range(len(qij_candid)):
#         for j in range(len(qij_candid)):
#             dist_Rn_R[i, j] = 2 * np.nan_to_num(np.arccos(np.abs(qij_candid[i] @ qij_candid[j])))
# 
#     dist = np.sum(np.log(1 + np.power(dist_Rn_R, 2)), axis=1)
#     dist = np.nan_to_num(dist, nan=np.inf)
#     dist[dist == 0] = np.inf
#     R_star_ind = np.argmin(dist)
#     Rij_star = Rij_candid[R_star_ind]
#     return Rij_star
# =============================================================================

def find_R_optim(Rij_candid):
    """
    Description
    -------
    Returns optimal rotation selected from the hypothesis. Follows latent ransac
    in which it is selected the one most representative of all. Eq (8) from paper

    Parameters
    ----------
    Rij_candid : numpy.ndarray
        Array with all rotation hypothesis.

    Returns
    -------
    Rij_star : numpy.ndarray
        Optimal relative rotation following latent ransac criteria.

    """
    qij_candid = np.array([rotration_as_quat(Rij) for Rij in Rij_candid])
    dist_Rn_R = np.nan_to_num(np.arccos(np.abs(cdist(qij_candid, qij_candid, lambda u, v: u @ v)))) * 2
    dist = np.sum(np.log(1 + np.power(dist_Rn_R, 2)), axis=1)
    dist = np.nan_to_num(dist, nan=np.inf)
    dist[dist == 0] = np.inf
    R_star_ind = np.argmin(dist)
    Rij_star = Rij_candid[R_star_ind]
    return Rij_star



def plot_cams(P_list, s=0.03, fig=None):
    """
    Description
    -------
    Given a list of Camera objects, plot camera sketches with camera center
    and rotation

    Parameters
    ----------
    P_list : list
        List of camera objects.
    s : float, optional
        Scale size of camera sketckes. The default is 0.03.
    fig : matplotlib.pyplot.figure(), optional
        If given, cameras are drawn in the figure fig. The default is None.

    Returns
    -------
    None.

    """
    Rs = []
    ts = []
    for P in P_list:
        Rs.append(P.R.T)
        ts.append(-P.R.T @ P.t)
    
    std_c = s*np.array([[0,0,0],
                    [1,1,1],
                    [-1,1,1],
                    [-1,-1,1],
                    [1,-1,1]])
    
    if not fig: fig = plt.figure()
    
    
    for Ri, ti in zip(Rs,ts):
        c = np.random.rand(3,)    
        Ri = np.row_stack((Ri,np.zeros(3)))
        ti = np.concatenate((ti.reshape((3,1)), np.ones((1,1))))
        T = np.concatenate((Ri, ti), axis=1)
        
        transf_c = np.concatenate((std_c, np.ones((5,1))), axis=1)
        transf_c = T @ transf_c.T
        transf_c /= transf_c[3,:]
        transf_c = transf_c.T
        
        std_l = []
        std_l.append([transf_c[0], transf_c[1]])
        std_l.append([transf_c[0], transf_c[2]])
        std_l.append([transf_c[0], transf_c[3]])
        std_l.append([transf_c[0], transf_c[4]])
        std_l.append([transf_c[1], transf_c[2]])
        std_l.append([transf_c[2], transf_c[3]])
        std_l.append([transf_c[3], transf_c[4]])
        std_l.append([transf_c[4], transf_c[1]])
        
        #fig = plt.figure()
        ax = fig.gca(projection='3d')
        for l in std_l:
            ax.plot([l[0][0], l[1][0]],
                    [l[0][1], l[1][1]],
                    [l[0][2], l[1][2]], c=c)
        plt.axis('on')
    
def plot_cam(P, s=0.03, c="R", fig=None):
    """
    Description
    -------
    Given a Camera object, it plots in 3D a camera sketch with 
    camera center location and rotation.

    Parameters
    ----------
    P : Camera
        Camera object to be ploted.
    s : float, optional
        Scale for camera sketch size. The default is 0.03.
    c : str, optional
        String code for color. The default is "R".
    fig : matplotlib.pyplot.figure(), optional
        Pyplot figure. If it is given, a camamera is drawn over this plot.
        The default is None.

    Returns
    -------
    None.

    """
    Rs = P.R.T
    ts = -P.R.T @ P.t
    std_c = s*np.array([[0,0,0],
                    [1,1,1],
                    [-1,1,1],
                    [-1,-1,1],
                    [1,-1,1]])
            
    Rs = np.row_stack((Rs,np.zeros(3)))
    ts = np.concatenate((ts.reshape((3,1)), np.ones((1,1))))
    T = np.concatenate((Rs, ts), axis=1)
    
    transf_c = np.concatenate((std_c, np.ones((5,1))), axis=1)
    transf_c = T @ transf_c.T
    transf_c /= transf_c[3,:]
    transf_c = transf_c.T
    
    std_l = []
    std_l.append([transf_c[0], transf_c[1]])
    std_l.append([transf_c[0], transf_c[2]])
    std_l.append([transf_c[0], transf_c[3]])
    std_l.append([transf_c[0], transf_c[4]])
    std_l.append([transf_c[1], transf_c[2]])
    std_l.append([transf_c[2], transf_c[3]])
    std_l.append([transf_c[3], transf_c[4]])
    std_l.append([transf_c[4], transf_c[1]])
        
    if not fig: fig = plt.figure()
    ax = fig.gca(projection='3d')
    for l in std_l:
        ax.plot([l[0][0], l[1][0]],
                [l[0][1], l[1][1]],
                [l[0][2], l[1][2]], c=c)
    plt.axis('on')
    

def verify_duplet(domain, view0, view1, F, threshold, tri_match = None):
    """
    Description
    -------
    #Given the initial view0 and second view1, F (according v0 and v1)
    # domain and optional tri_match, it checks matches in v0 and v1 that meet
    # hofer (2013) criteria based on line-endpoints and their intersection with
    #epipolar lines.
    #computes the the intersection of line segments in view i+1 with epipolar lines.
    #This function returns inliers following hofer(2013) paper (we do more relaxed version)
    #as we got few triplet matches.
    #the intersection of the line segments can be considered as the correction of 
    #the lines endpoints. It could be beneficial if 2 views triangulation is performed
    #This function is similar to find_scores_posesModel from poses_ransac.py file
    
    Parameters
    ----------
    domain : Domain
        Domain object with SfM information.
    view0 : int
        id view 0.
    view1 : int
        id view 1.
    F : numpy.ndarray
        Fundamental matrix cameras 0-1.
    threshold : float
        Fraction of line segment of distances between intersections with
        epipolar line to classify line segments as inliers or outliers.
    tri_match : bool, optional
        If true, the verification is made with matches in 3 views. The default is None.

    Returns
    -------
    num_inl : int
        number of inliers.
    l_matches_inliers : numpy.ndarray
        Ids of line matches that met the criteria.

    """

    
    #array with the 2 view matches from tri_match given or directly from domain line matches
    if tri_match is not None:
        l_matches = tri_match[:,[view0,view1]]
    else:        
        if view0>view1:#to flip matches in case of analize 2-0 instead of 0-2. As matcher gives the order 0-2
            l_matches = np.array(domain.line_match[str(view1)+str(view0)][1])    
            l_matches = l_matches[:,[1,0]] 
        else:
            l_matches = np.array(domain.line_match[str(view0)+str(view1)][1])
    #endpts_v0 = [] #end points correspondences v0
    endpts_v1 = [] #end points correspondences v1, meet with criterias
    endpts_v1_int = [] #end points correspondences corrected v1 (intersections), meet with criterias
    endpts_v1_full = [] #end points correspondences v1 full data set
    endpts_v1_int_full = [] #end points correspondences corrected v1 (intersections) full data set
    pr = 0
    num_inl = 0 #Number of inliers to give score 1
    l_matches_inliers = [] #List with the inlier matches
    ind_l_matches_inliers = [] #List with the index from tri_match of inlier matches 
            
    
    #for lm in l_matches[1]:
    for ind, lm in enumerate(l_matches):
        #if pr%100==0:
            #print("progress correcting endpoints: ", np.round(100*pr/len(l_matches[1]), 2), "%")
        
        #print("Correcting endpoints and computing inliers based on epipolar lines---------------------")
        
        pr+=1
        l_i = domain.lines2d[lm[0]]
        l_i1 = domain.lines2d[lm[1]]
        
        #Making endpoints homogeneus
        x_i_0 = conv_pt_homogeneus(l_i.points[0].coord)
        x_i_1 = conv_pt_homogeneus(l_i.points[1].coord)
        x_i1_0 = conv_pt_homogeneus(l_i1.points[0].coord)
        x_i1_1 = conv_pt_homogeneus(l_i1.points[1].coord)
        
        #Finding line representation for segment in view i1
        line_i1 = line_parametrization(x_i1_0, x_i1_1)        
        #Epipolar lines: v0 over v1
        line0 = np.dot(F, x_i_0)
        line1 = np.dot(F, x_i_1)
        
        #Finding interseccions between line segment and epipolar lines
        xi0,yi0 = intersection(line_i1, line0)
        xi1,yi1 = intersection(line_i1, line1)
        
        #Creating full data set of endpoints and intersections
        endpts_v1_full.append(l_i1.points[0].coord)
        endpts_v1_full.append(l_i1.points[1].coord)
        endpts_v1_int_full.append([xi0,yi0])
        endpts_v1_int_full.append([xi1,yi1])
        
        #Thresold to the distance between endpoints (x0,x1) and corrected ones (xp,xq). 
        #1) Distance of at least one end point has to be is lower than 10% of the min(||xp-xq||, ||x0-x1||)
        #2) the second endpoint has to be closer to the other intersection point. 
        len_line = np.linalg.norm(np.array([xi1[0],yi1[0]])-np.array([xi0[0],yi0[0]]))
        len_line_corr = np.linalg.norm(np.array(l_i1.points[1].coord)-np.array(l_i1.points[0].coord))
        #thr = 0.01*min(len_line,len_line_corr)
        thr = threshold*min(len_line,len_line_corr)
        
        #Checking if match meets the threshold (disti_j: distance endpoint i to inters j)
        dist1_1 = np.linalg.norm(np.array(l_i1.points[0].coord)-np.array([xi0[0],yi0[0]]))
        dist2_2 = np.linalg.norm(np.array(l_i1.points[1].coord)-np.array([xi1[0],yi1[0]]))
        
        dist1_2 = np.linalg.norm(np.array(l_i1.points[0].coord)-np.array([xi1[0],yi1[0]]))
        dist2_1 = np.linalg.norm(np.array(l_i1.points[1].coord)-np.array([xi0[0],yi0[0]]))
        
        if dist1_1<thr or dist2_2 < thr: #Criteria 1)
            
            if (dist1_1<=dist2_2 and dist2_2<dist2_1) or (dist2_2<dist1_1 and dist1_1<dist1_2): #Criteria 2
        
                l_matches_inliers.append(lm)
                ind_l_matches_inliers.append(ind)
                num_inl+=1
                
                #endpts_v0.append(l_i.points[0].coord)
                #endpts_v0.append(l_i.points[1].coord)
                endpts_v1.append(l_i1.points[0].coord)
                endpts_v1.append(l_i1.points[1].coord)
                endpts_v1_int.append([xi0,yi0])
                endpts_v1_int.append([xi1,yi1])                
      
    #Making arrays of endpoints and intersections filtered and full data
    endpts_v1 = np.array(endpts_v1).reshape((-1,2))
    endpts_v1_int = np.array(endpts_v1_int).reshape((-1,2))
    endpts_v1_full = np.array(endpts_v1_full).reshape((-1,2))
    endpts_v1_int_full = np.array(endpts_v1_int_full).reshape((-1,2))
        
    #Distances from inliers to give score 2
    #distances = np.linalg.norm(endpts_v1[0::2]-endpts_v1_int[0::2], axis=1) + np.linalg.norm(endpts_v1[1::2]-endpts_v1_int[1::2], axis=1)
    #Distances all data set (endpoints to intersections)
    #distances_full = np.linalg.norm(endpts_v1_full[0::2]-endpts_v1_int_full[0::2], axis=1) + np.linalg.norm(endpts_v1_full[1::2]-endpts_v1_int_full[1::2], axis=1)
    
    l_matches_inliers = np.array(l_matches_inliers)
    
    return  num_inl, l_matches_inliers

def verify_triplet(domain, ransac_fit, view0, view1, view2, threshold, tri_match = None):
    """
    Description
    -------
    #This function is similar to get_error3 from poses_ransac.py file
    #The aim is to find the triplet matches that met the hofer2013 criteria
    #With the original value of the paper (0.01*min(line_length, intersecctions_length))
    #It did not show nice results for ransac as there are few triplet matches
    #then it will be relaxed the constrain.
    #Here it is just evaluated correspondences in the 3 views and counting inliers
    #when it met the criteria three times 1)views 0-1, 2) views 1-2, views 2-0  
    #(or twice maybe to be softer)
    
    Parameters
    ----------
    
    domain : Domain
        Domain object with SfM information.
    ransac_fit : List
        List with camera models gotten from ransac fitting.
    view0 : int
        id view 0.
    view1 : int
        id view 1.
    view2 : int
        id view 2.
    threshold : float
        Fraction of line segment of distances between intersections with
        epipolar line to classify line segments as inliers or outliers.
    tri_match : bool, optional
        If true, the verification is made with matches in 3 views. The default is None.

    Returns
    -------
    num_inl : int
        number of inliers.
    l_matches_inliers : numpy.ndarray
        Ids of line matches that met the criteria.

    """

    
    P0, P1, P2 = ransac_fit[0], ransac_fit[1], ransac_fit[2]
    
    #Finding fundamental matrices
    rR0, rt0 = relativeCameraMotion(P0, P1)
    F0 = F_from_KRt2(domain.K, rR0, rt0)
    rR1, rt1 = relativeCameraMotion(P1, P2)
    F1 = F_from_KRt2(domain.K, rR1, rt1)
    rR2, rt2 = relativeCameraMotion(P2, P0) #could be the other way around (P0,P2)
    F2 = F_from_KRt2(domain.K, rR2, rt2)
    
    #triplet correspondances
    if tri_match is None:
        tri_match = find_correspondences_3v(domain, view0, view1, view2, n_corr="full")
    #views 0-1
    num_inl0, l_matches_inliers0 = verify_duplet(domain, view0, view1, F0, threshold, tri_match=tri_match)
    #views 1-2
    num_inl1, l_matches_inliers1 = verify_duplet(domain, view1, view2, F1, threshold, tri_match=tri_match)
    #views 2-0
    num_inl2, l_matches_inliers2 = verify_duplet(domain, view2, view0, F2, threshold, tri_match=tri_match)
    
    #Checking inliers for the triplet
    l_matches_inliers0 = np.array(l_matches_inliers0)
    l_matches_inliers1 = np.array(l_matches_inliers1)
    l_matches_inliers2 = np.array(l_matches_inliers2)
    l_matches_inliers = []
    
    for i, l0 in enumerate(l_matches_inliers0):
        if (l0[1]==l_matches_inliers1[:,0]).sum()==1:
            j = np.where((l0[1]==l_matches_inliers1[:,0]))[0][0]
            if (l_matches_inliers1[j][1]==l_matches_inliers2[:,0]).sum()==1:
                l_matches_inliers.append(tri_match[i])  
    
    
    #print(distances0, distances1)
    num_inl = len(l_matches_inliers)
    l_matches_inliers = np.array(l_matches_inliers)
            
    return num_inl, l_matches_inliers

def plot_triplet_matches(domain, view0, view1, view2, tri_match):
    """
    Description
    --------
    Given domain, views ids, and tri_match, it will plot triple_matches 

    Parameters
    ----------
    domain : Domain
        Domain object with information of SfM.
    view0 : int
        id View 0.
    view1 : int
        id View 1.
    view2 : int
        id View 2.
    tri_match : numpy.ndarray
        ids of matches in three views.

    Returns
    -------
    None.

    """
    
      
    #Reading images
    im0 = domain.views[view0].image
    im1 = domain.views[view1].image
    im2 = domain.views[view2].image
    #im1 = domain.views[view_id].image
    #im2 = domain.views[view_id+1].image
    #Creating side by side image
    sh0 = im0.shape
    sh1 = im1.shape
    sh2 = im2.shape
    shift_x0 = sh0[1]
    shift_x1 = sh0[1]+sh1[1]
    im = np.zeros((max(sh0[0],sh1[0],sh2[0]), sh0[1]+sh1[1]+sh2[1], 3))
    im[0:sh0[0], 0:sh0[1], :] = im0
    im[0:sh1[0], sh0[1]:sh0[1]+sh1[1], :] = im1
    im[0:sh2[0], sh0[1]+sh1[1]:, :] = im2
    im = im.astype('uint8')
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #Ploting
    plt.figure()
    for l in tri_match:
        c = np.random.rand(3,)
        #line view0
        x = [domain.lines2d[l[0]].points[0].coord[0], domain.lines2d[l[0]].points[1].coord[0]]
        y = [domain.lines2d[l[0]].points[0].coord[1], domain.lines2d[l[0]].points[1].coord[1]]
        plt.plot(x, y, color=c, linewidth=3)
        #plt.annotate(str(self.lines2d[l[0]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")        
        #line view1
        x = [domain.lines2d[l[1]].points[0].coord[0]+shift_x0, domain.lines2d[l[1]].points[1].coord[0]+shift_x0]
        y = [domain.lines2d[l[1]].points[0].coord[1], domain.lines2d[l[1]].points[1].coord[1]]
        #print(x,y)
        plt.plot(x, y, color=c, linewidth=3)
        #plt.annotate(str(self.lines2d[l[0]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")
        #line view i+1
        x = [domain.lines2d[l[2]].points[0].coord[0]+shift_x1, domain.lines2d[l[2]].points[1].coord[0]+shift_x1]
        y = [domain.lines2d[l[2]].points[0].coord[1], domain.lines2d[l[2]].points[1].coord[1]]
        #print(x,y)
        plt.plot(x, y, color=c, linewidth=3)
        #plt.annotate(str(self.lines2d[l[1]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")
    plt.imshow(im, cmap='gray')
    plt.savefig("../../output/"+domain.views[view0].name[:-4]+"_"+domain.views[view1].name[:-4]+"_"+domain.views[view2].name[:-4]+"_triplet_matcher.png", dpi=1000)

def plot_duplet_matches(domain, view0, view1, dup_match):
    """
    Description
    ---------
    Given domain, views ids, and tri_match, it will plot triple_matches 

    Parameters
    ----------
    domain : Domain
        Domain object with information of SfM.
    view0 : int
        id View 0.
    view1 : int
        id View 1.
    dup_match : numpy.ndarray
        ids of matches in two views.

    Returns
    -------
    None.

    """
      
    #Reading images
    im0 = domain.views[view0].image
    im1 = domain.views[view1].image
    
    #Creating side by side image
    sh0 = im0.shape
    sh1 = im1.shape
    
    shift_x0 = sh0[1]
    
    im = np.zeros((max(sh0[0],sh1[0]), sh0[1]+sh1[1], 3))
    im[0:sh0[0], 0:sh0[1], :] = im0
    im[0:sh1[0], sh0[1]:sh0[1]+sh1[1], :] = im1
    
    im = im.astype('uint8')
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #Ploting
    plt.figure()
    for l in dup_match:
        
        c = np.random.rand(3,)
        #line view0
        x = [domain.lines2d[int(l[0])].points[0].coord[0], domain.lines2d[int(l[0])].points[1].coord[0]]
        y = [domain.lines2d[int(l[0])].points[0].coord[1], domain.lines2d[int(l[0])].points[1].coord[1]]
        plt.plot(x, y, color=c, linewidth=3)
        #plt.annotate(str(self.lines2d[l[0]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")        
        #line view1
        x = [domain.lines2d[int(l[1])].points[0].coord[0]+shift_x0, domain.lines2d[int(l[1])].points[1].coord[0]+shift_x0]
        y = [domain.lines2d[int(l[1])].points[0].coord[1], domain.lines2d[int(l[1])].points[1].coord[1]]
        #print(x,y)
        plt.plot(x, y, color=c, linewidth=3)
        #plt.annotate(str(self.lines2d[l[0]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")
        #plt.annotate(str(self.lines2d[l[1]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")
    plt.imshow(im, cmap='gray')
    plt.savefig("../../output/"+domain.views[view0].name[:-4]+"_"+domain.views[view1].name[:-4]+"_duplet_matcher.png", dpi=1000)