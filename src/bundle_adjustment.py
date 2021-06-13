#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:48:20 2021
@author: pantoja
"""

# Bundle implementation based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

# It is a modified version. The residual function fun is calculated with
# the projection of image points as PX. Here are optimize 12 parameters, 9 from
# rotation matrix R and 3 from translation t. (P = [R|t] with coordinates
# normalize by inv(K))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
import cv2


def project(points, camera_params_):
    """
    Description
    ------
    Convert 3-D points to 2-D spherical representation p_ by projecting onto images.

    Parameters
    ----------
    points : numpy.ndarray
        points to be projected.
    camera_params_ : numpy.ndarray
        Camera parameters used to triangulate.

    Returns
    -------
    points_2d_p_ : numpy.ndarray
        Projected points in spherical coordinates.

    """

    if camera_params_[0].shape[1] == 4:
        R = camera_params_[:, :, :3]
        c = camera_params_[:, :, 3]
    else:
        R = camera_params_[:, :, 0]
        R = np.array([cv2.Rodrigues(Ri)[0] for Ri in R])
        c = camera_params_[:, :, 1]

    c = c.reshape((len(c), 3, 1))
    t = np.array([np.dot(R[i], -c[i]) for i in range(len(R))])
    # print(t.shape)
    P = np.concatenate((R, t), axis=2)

    # P = camera_params #using t instead of c to optimize
    X = points
    X = np.concatenate((X, np.ones((len(X), 1))), axis=1)
    X = X.T

    points_proj = np.empty(shape=[3, 0])
    for i in range(len(X.T)):
        points_proj = np.concatenate((points_proj, np.dot(P[i], X[:, i].reshape((4, 1)))), axis=1)

    points_proj = points_proj / np.linalg.norm(points_proj, axis=0)

    # for i in range(3):
    #        points_proj[i] /= points_proj[2]

    points_2d_p_ = points_proj.T  # approximation

    # print(points_proj)

    return points_2d_p_


# function which returns a vector of residuals (fun).
def fun(params, n_cameras, n_points, camera_indices_pt, pt_indices, points_2d_p, lines_2d):
    """
    Description
    -------    
    Compute residuals.
    

    Parameters
    ----------
    params : numpy.ndarray
        params contains camera parameters and 3-D coordinates..
    n_cameras : int
        number of cameras in BA.
    n_points : int
        number of end points in BA.
    camera_indices_pt : numpy.ndarrya
        For each 2d point assign the camera id that is influenced by the point.
    pt_indices : numpy.ndarrya
        For each 2d point assign the 3D point id that is influenced by the point.
    points_2d_p : numpy.ndarray
        Projected 2d points in spherical coordinates.
    lines_2d: list 
        lines in 2d.

    Returns
    -------
    numpy.ndarray
        Residuals calculated for each 2d line.

    """

    n_cam_par = int((len(params) - 3 * n_points) / (n_cameras))  # FROM (**) in def run_bundle_adjustment

    camera_params = params[:n_cameras * n_cam_par].reshape((n_cameras, 3, int(n_cam_par / 3)))
    points_3d = params[n_cameras * n_cam_par:].reshape((n_points, 3))

    points_2d_p_ = project(points_3d[pt_indices], camera_params[camera_indices_pt])

    # Loss function adapted from paper
    residuals = []
    for i, l in enumerate(lines_2d):
        n_ji = l.n.reshape((3, 1))
        lamb = 2
        # loss endpoint 0
        # p0 = points_2d_p[2*i].reshape((3,1))
        p0 = l.points[0].p.reshape((3, 1))
        p_0 = points_2d_p_[2 * i].reshape((3, 1))
        e0 = ((lamb * np.dot(p_0.T, n_ji))[0][0] ** 2) + (
        ((np.dot(p_0.T, np.cross(n_ji.ravel(), p0.ravel()).reshape((3, 1)))[0][0]) ** 2))

        # loss endpoint 1
        # p1 = points_2d_p[2*i+1].reshape((3,1))
        p1 = l.points[1].p.reshape((3, 1))
        p_1 = points_2d_p_[2 * i + 1].reshape((3, 1))
        e1 = ((lamb * np.dot(p_1.T, n_ji))[0][0] ** 2) + (
        ((np.dot(p_1.T, np.cross(n_ji.ravel(), p1.ravel()).reshape((3, 1)))[0][0]) ** 2))

        residuals.append(e0 + e1)

    residuals = np.array(residuals)

    return (residuals).ravel()


# Jacobian of fun is cumbersome. Then applied Finite difference approximation.
# it is provided Jacobian sparsity structure
def bundle_adjustment_sparsity(n_cameras, n_points, n, m, camera_indices, line_indices):
    """
    Description
    --------
    Sparce matrix used in least-squares method from scipy

    Parameters
    ----------
    n_cameras : int
        number of camaras in BA.
    n_points : int
        number of 3D points in BA.
    n : int
        dim 1 of sparce matrix.
    m : int
        dim 1 of sparce matrix..
    camera_indices : numpy.ndarray
        camera indices influenced by a 2d line.
    line_indices : numpy.ndarray
        3D lines indices influenced by a 2d line.

    Returns
    -------
    A : numpy.ndarray
        Sparce matrix used to compute Jacobian inside scipy least-square algorithm.

    """
    # m = camera_indices.size * 2
    # n = n_cameras * 12 + n_points * 3

    n_cam_par = int((n - 3 * n_points) / (n_cameras))  # FROM (**) in def run_bundle_adjustment

    A = lil_matrix((m, n),
                   dtype=int)  # in tradicional, 2 rows by 2dpoint, here 1 row by 2dline as loss function is given by line

    i = np.arange(camera_indices.size)

    for s in range(n_cam_par):  # filling elements that relates lines2d with camera params
        # A[2 * i, camera_indices* n_cam_par + s] = 1
        # A[2 * i + 1, camera_indices * n_cam_par + s] = 1
        A[i, camera_indices * n_cam_par + s] = 1

    for s in range(3):  ##filling elements that relates lines2d with 3D point coordinates
        # A[2 * i, n_cameras * n_cam_par + line_indices * 3 + s] = 1
        # A[2 * i + 1, n_cameras * n_cam_par + line_indices * 3 + s] = 1
        A[i, n_cameras * n_cam_par + line_indices * 3 + s] = 1

    return A


def point_wise_input(lines_3d, lines_2d, camera_indices, line_indices):
    """
    Description
    -------
    # This function find the point wise version of the input for BA

    Parameters
    ----------
    lines_3d : list
        List of 3D lines.
    lines_2d : numpy,ndarray
        lines 2d.
    camera_indices : TYPE
        DESCRIPTION.
    line_indices : TYPE
        DESCRIPTION.

    Returns
    -------
    points_3d : TYPE
        DESCRIPTION.
    points_2d_p : TYPE
        DESCRIPTION.
    camera_indices : numpy.ndarray
        camera indices influenced by a 2d line.
    pt_indices : TYPE
        3D points indices influenced by a 2d line..

    """
    

    points_3d = []  # np.empty(shape=[0,3])
    points_2d_p = []
    # points_2d = [] #np.empty(shape=[0,2])
    camera_indices_pt = []
    pt_indices = []

    for L in lines_3d:
        points_3d.append(L.points[0].coord[:3])
        points_3d.append(L.points[1].coord[:3])

    for i, l in enumerate(lines_2d):
        # points_2d.append(l.points[0].coord)
        points_2d_p.append(l.points[0].p)
        camera_indices_pt.append(camera_indices[i])
        pt_indices.append(2 * line_indices[i])

        points_2d_p.append(l.points[1].p)
        camera_indices_pt.append(camera_indices[i])
        pt_indices.append(2 * line_indices[i] + 1)

    points_3d = np.array(points_3d)
    points_2d_p = np.array(points_2d_p)
    camera_indices_pt = np.array(camera_indices_pt)
    pt_indices = np.array(pt_indices)

    return points_3d, points_2d_p, camera_indices_pt, pt_indices


# Run bundle adjustment
def run_bundle_adjustment(input_bundle_adjustment, loss_f):
    """
    Description
    -------
    Run bundle adjustment using least squares function from scipy library

    Parameters
    ----------
    input_bundle_adjustment : dict
        Dictionary containing information from current SfM Domain.
    loss_f : str
        Loss function used in least-squares.

    Returns
    -------
    op_camera_params : numpy.ndarray
        Camera parameters after BA.
    op_points_3d : numpy.ndarray
        3D points after BA.
    residual : numpy.ndarray
        residuals for lines2d after BA.

    """
    print("--------------------RUNING BUNDLE ADJUSTMENT ---------------------------")

    camera_params = input_bundle_adjustment['camera_params']
    lines_3d = input_bundle_adjustment['Lines_3d']
    camera_indices = input_bundle_adjustment['camera_indices']
    line_indices = input_bundle_adjustment['lines_indices']
    lines_2d = input_bundle_adjustment['lines_2d']

    # points_2d_p = input_bundle_adjustment['points_2d']
    # lines_n = input_bundle_adjustment['lines_n']

    # Getting point wise version to help projection and residual functions
    points_3d, points_2d_p, camera_indices_pt, pt_indices = point_wise_input(lines_3d, lines_2d, camera_indices,
                                                                             line_indices)

    # Problem numbers
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n_lines = len(lines_3d)

    n = camera_params.shape[1] * n_cameras + 3 * n_points  # (**)
    # m = 2 * points_2d.shape[0]
    # m = 3 * points_2d_p.shape[0] #as instead of 2d coord, it is p representation -->maybe this worng as fun needs to output the vector of residuals
    # m = 1 * points_2d_p.shape[0] #as instead of 2d coord, it is p representation, just 1 variable
    m = 1 * len(lines_2d)  # as instead of 2d coord, it is p representation, just 1 variable

    print("n_cameras: {}".format(n_cameras))
    print("n_points3D: {}".format(n_points))
    print("n_lines3D: {}".format(n_lines))
    print("Total number of parameters: {}".format(n))
    # print("Total number of residuals: {}".format(m))
    # print("Total number of residuals: {}".format(m/3)) #m/3 as a residual is computed with the 3 elements of p
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d_p, lines_n)
    f0 = fun(x0, n_cameras, n_points, camera_indices_pt, pt_indices, points_2d_p, lines_2d)
    plt.figure()
    plt.plot(f0)
    A = bundle_adjustment_sparsity(n_cameras, n_points, n, m, camera_indices, line_indices)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', loss=loss_f, \
                        args=(n_cameras, n_points, camera_indices_pt, pt_indices, points_2d_p, lines_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    plt.plot(res.fun)

    # updating camera parameters and 3d points
    op_camera_params = res.x[:camera_params.shape[1] * n_cameras]
    op_camera_params = op_camera_params.reshape((n_cameras, camera_params.shape[1]))
    op_points_3d = res.x[camera_params.shape[1] * n_cameras:]
    op_points_3d = op_points_3d.reshape((n_points, 3))

    # Ressiduals variable
    residual = np.copy(res.fun)

    # ploting 3d points
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(op_points_3d[:, 0], op_points_3d[:, 1], op_points_3d[:, 2], 'k.')
    # plt.axis('off')
    #
    # # Ploting lines
    # for i in range(int(len(op_points_3d) / 2)):
    #     line_start = np.array([op_points_3d[2 * i][0], op_points_3d[2 * i][1], op_points_3d[2 * i][2]])
    #     line_end = np.array([op_points_3d[2 * i + 1][0], op_points_3d[2 * i + 1][1], op_points_3d[2 * i + 1][2]])
    #     ax.plot([line_start[0], line_end[0]],
    #             [line_start[1], line_end[1]],
    #             [line_start[2], line_end[2]], c='blue')
    # plt.axis('on')

    return op_camera_params, op_points_3d, residual