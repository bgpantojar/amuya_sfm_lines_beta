#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:13:54 2021
@author: pantoja
"""

import numpy as np
import scipy
from utils import *
from camera import Camera


class PosesModel:  # (object):
    """
    
    Description
    -------
        Class used to run ransac algorithm.
        This class uses the eq [9] from paper to find t2 and t3. With those
        camera matrices are found and then the Fundamental Matrix.
        Error is calculated comparing line-segment end points in views 1 and 2
        and their intersections with epipolar lines
    """

    def __init__(self, domain, K, R0, R1, R2, view0=0, view1=1, view2=2, debug=False):
        """
        Description
        ------
        This generates a ransac object from class PosesMoel

        Parameters
        ----------
        domain : Domain
            Object from class Domain containing information about structure and motion.
        K : numpy.ndarray
            Matrix with intrinsic parameters of the camera.
        R0 : numpy.ndarray
            Rotation matrix for camera 0.
        R1 : numpy.ndarray
            Rotation matrix for camera .
        R2 : numpy.ndarray
            Rotation matrix for camera 2.
        view0 : int, optional
            view0 id. The default is 0.
        view1 : int, optional
            view1 id. The default is 1.
        view2 : int, optional
            view2 id. The default is 2.
        debug : bool, optional
            print warnings for debuging. The default is False.

        Returns
        -------
        None.

        """
        self.domain = domain
        self.K = K
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        # self.corr_3v = corr_3v
        self.view0 = view0
        self.view1 = view1
        self.view2 = view2
        self.debug = debug

    def fit(self, data):
        """
        Solves null space of Eq (9) to find translations of cameras 1 and 2, 
        and creates camera objects for those.

        Parameters
        ----------
        data : list
            List with the ids of 5 line correspondences in 3 views.

        Returns
        -------
        list
            List of Camera objects for views 0,1,2.

        """

        A = find_matrix_A(self.domain, self.R0, self.R1, self.R2, data)
        A = A / scipy.linalg.norm(A)  # normalizing matrix
        x = scipy.linalg.null_space(A)

        if x.shape[1] > 1:  # avoid null space multidimensional
            print("Multidimensional null space!")
            return None

        else:

            X = x[:-6]
            zz = X[2::3]
            # print(zz)
            if sum(zz[:, 0] < 0) == 5:
                x = -x
            # t1 = x[-6:-3]
            # t2 = x[-3:]
            X = x[:-6]
            zz = X[2::3]
            print(zz)
            if sum(zz[:, 0] > 0) < 5:  # avoid mixed signs in z
                return None
            else:
                # print(zz)
                # Creating Projective Camera Matrices
                P0 = Camera(self.K @ np.hstack((self.R0, np.zeros((3, 1)))))
                P0.factor()
                P0.center()
                #print(x[-6:-3])
                P1 = Camera(self.K @ np.hstack((self.R1, x[-6:-3].reshape((3, 1)))))
                P1.factor()
                P1.center()
                P2 = Camera(self.K @ np.hstack((self.R2, x[-3:].reshape((3, 1)))))
                P2.factor()
                P2.center()

                return [P0, P1, P2]

    def get_error(self, model):
        """
        Computes error for the predicted model pairwise views 0-1,1-2,2-0,. It is based on paper of
        Hofer et al. (2013). Epipolar lines from line-endpoints in view i are created in
        view i+1. Endpoints correspondences in view i+1 should lay in those epipolar lines.
        They will not lay exactly, producin a distance error here calculated

        Parameters
        ----------
        model : list
            List of Camera objects for cameras 0,1,2.

        Returns
        -------
        error : float
            mean distances from line endpoints in views i+1 to epipolar lines.
        error_inl : list
            List with ids of lines that are inliers. Distances lower than a threshold.
        num_inl : int
            number of inliers.

        """

        #Read cameras calculated with fit
        P0, P1, P2 = model[0], model[1], model[2]

        #Compute scores for pair 0-1
        rR0, rt0 = relativeCameraMotion(P0, P1) #relative rotation and translation based on cameras
        F0 = F_from_KRt2(self.K, rR0, rt0) #fundamental matrix based on relative rotation and translation
        distances0, distances_full0, num_inl0, l_matches_inliers0, ind_inl0 = self.find_scores_posesModel(self.domain,
                                                                                                          self.view0,
                                                                                                          self.view1,
                                                                                                          F=F0)

        #Compute scores for pair 1-2
        rR1, rt1 = relativeCameraMotion(P1, P2)#relative rotation and translation based on cameras
        F1 = F_from_KRt2(self.K, rR1, rt1) #fundamental matrix based on relative rotation and translation
        distances1, distances_full1, num_inl1, l_matches_inliers1, ind_inl1 = self.find_scores_posesModel(self.domain,
                                                                                                          self.view1,
                                                                                                          self.view2,
                                                                                                          F=F1)

        # Closing the triplet. Here the relative order is 0-2, maybe change it to 2-0
        # Compute scores for pair 1-2
        # rR2, rt2 = relativeCameraMotion(P0, P2)
        # rR2, rt2 = relativeCameraMotion(P2, P0)
        # F2 = F_from_KRt2(self.K, rR2, rt2)
        # distances2, distances_full2, num_inl2, l_matches_inliers2, ind_inl2 = self.find_scores_posesModel(self.domain, self.view0, self.view2, F=F2)
        # distances2, distances_full2, num_inl2, l_matches_inliers2, ind_inl2 = self.find_scores_posesModel(self.domain,
        #                                                                                                   self.view2,
        #                                                                                                   self.view0,
        #                                                                                                   F=F2)

        # print(distances0, distances1)
        error_inl = (np.mean(distances0) + np.mean(distances1)) / 2
        error = (np.mean(distances_full0) + np.mean(distances_full1)) / 2
        num_inl = num_inl0 + num_inl1

        # error_inl = (np.mean(distances0) + np.mean(distances1) + np.mean(distances2)) / 3
        # error = (np.mean(distances_full0) + np.mean(distances_full1) + np.mean(distances_full2)) / 3
        # num_inl = num_inl0 + num_inl1 + num_inl2
        # num_inl_ = len(l_matches_inliers0) + len(l_matches_inliers1)
        # num_inl__ = len(distances0)+len(distances1)
        # print("num inliers 2 ways: ", num_inl, ", ", num_inl_, ", ", num_inl__)

        return error, error_inl, num_inl

    def find_scores_posesModel(self, domain, view0, view1, F, tri_match=None):
        """
        Description
        ------
        Given the initial view i, F and domain, it computes the the
        intersection of line segments in view i+1 with epipolar lines.
        This function returns number of inliers following hofer(2013) paper.
        Scores are given as number of inliers and the distances between endpoints
        and intersections with epipolar lines for inliers and total dataset

        Parameters
        ----------
        domain : Domain
            Object from class Domain containing information about structure and motion.
        view0 : int
            view0 id.
        view1 : int
            view1 id.
        F : numpy.ndarray
            Fundamental matrix for views 0-1.
        tri_match : bool, optional
            If true, it is used the correspondences from 3 views. The default is None.

        Returns
        -------
        distances : numpy.ndarray
            Array with distances of endpoinlines to epipolar lines for inliers.
        distances_full : numpy.ndarray
            Array with distances of endpoinlines to epipolar lines for all.
        num_inl : int
            number of inlier line segments.
        l_matches_inliers : list
            list with ids of matches that are inliers.
        ind_l_matches_inliers : list
            list with indices of matches that are inliers..

        """


        if tri_match is not None:
            l_matches = tri_match[:, [view0, view1]]
        else:
            if view0 > view1:  # to flip matches in case of analize 2-0 instead of 0-2. As matcher gives the order 0-2
                l_matches = np.array(domain.line_match[str(view1) + str(view0)][1])
                l_matches = l_matches[:, [1, 0]]
            else:
                l_matches = np.array(domain.line_match[str(view0) + str(view1)][1])
        # endpts_v0 = [] #end points correspondences v0
        endpts_v1 = []  # end points correspondences v1, meet with criterias
        endpts_v1_int = []  # end points correspondences corrected v1 (intersections), meet with criterias
        endpts_v1_full = []  # end points correspondences v1 full data set
        endpts_v1_int_full = []  # end points correspondences corrected v1 (intersections) full data set
        pr = 0
        num_inl = 0  # Number of inliers to give score 1
        l_matches_inliers = []  # List with the inlier matches
        ind_l_matches_inliers = []  # List with the index of inlier matches

        # for lm in l_matches[1]:
        for ind, lm in enumerate(l_matches):
            # if pr%100==0:
            # print("progress correcting endpoints: ", np.round(100*pr/len(l_matches[1]), 2), "%")

            # print("Correcting endpoints and computing inliers based on epipolar lines---------------------")

            pr += 1
            l_i = domain.lines2d[lm[0]]
            l_i1 = domain.lines2d[lm[1]]

            # Making endpoints homogeneus
            x_i_0 = conv_pt_homogeneus(l_i.points[0].coord)
            x_i_1 = conv_pt_homogeneus(l_i.points[1].coord)
            x_i1_0 = conv_pt_homogeneus(l_i1.points[0].coord)
            x_i1_1 = conv_pt_homogeneus(l_i1.points[1].coord)

            # Finding line representation for segment in view i1
            line_i1 = line_parametrization(x_i1_0, x_i1_1)
            # Epipolar lines: v0 over v1
            line0 = np.dot(F, x_i_0)
            line1 = np.dot(F, x_i_1)

            # Finding interseccions between line segment and epipolar lines
            xi0, yi0 = intersection(line_i1, line0)
            xi1, yi1 = intersection(line_i1, line1)

            # Creating full data set of endpoints and intersections
            endpts_v1_full.append(l_i1.points[0].coord)
            endpts_v1_full.append(l_i1.points[1].coord)
            endpts_v1_int_full.append([xi0, yi0])
            endpts_v1_int_full.append([xi1, yi1])

            # Thresold to the distance between endpoints (x0,x1) and corrected ones (xp,xq).
            # 1) Distance of at least one end point has to be is lower than 10% of the min(||xp-xq||, ||x0-x1||)
            # 2) the second endpoint has to be closer to the other intersection point.
            len_line = np.linalg.norm(np.array([xi1[0], yi1[0]]) - np.array([xi0[0], yi0[0]]))
            len_line_corr = np.linalg.norm(np.array(l_i1.points[1].coord) - np.array(l_i1.points[0].coord))
            # thr = 0.01*min(len_line,len_line_corr)
            thr = 0.01 * min(len_line, len_line_corr)

            # Checking if match meets the threshold (disti_j: distance endpoint i to inters j)
            dist1_1 = np.linalg.norm(np.array(l_i1.points[0].coord) - np.array([xi0[0], yi0[0]]))
            dist2_2 = np.linalg.norm(np.array(l_i1.points[1].coord) - np.array([xi1[0], yi1[0]]))

            dist1_2 = np.linalg.norm(np.array(l_i1.points[0].coord) - np.array([xi1[0], yi1[0]]))
            dist2_1 = np.linalg.norm(np.array(l_i1.points[1].coord) - np.array([xi0[0], yi0[0]]))

            if dist1_1 < thr or dist2_2 < thr:  # Criteria 1)

                if (dist1_1 <= dist2_2 and dist2_2 < dist2_1) or (
                        dist2_2 < dist1_1 and dist1_1 < dist1_2):  # Criteria 2

                    l_matches_inliers.append(lm)
                    ind_l_matches_inliers.append(ind)
                    num_inl += 1

                    # endpts_v0.append(l_i.points[0].coord)
                    # endpts_v0.append(l_i.points[1].coord)
                    endpts_v1.append(l_i1.points[0].coord)
                    endpts_v1.append(l_i1.points[1].coord)
                    endpts_v1_int.append([xi0, yi0])
                    endpts_v1_int.append([xi1, yi1])

                    # Making arrays of endpoints and intersections filtered and full data
        endpts_v1 = np.array(endpts_v1).reshape((-1, 2))
        endpts_v1_int = np.array(endpts_v1_int).reshape((-1, 2))
        endpts_v1_full = np.array(endpts_v1_full).reshape((-1, 2))
        endpts_v1_int_full = np.array(endpts_v1_int_full).reshape((-1, 2))

        # Distances from inliers to give score 2
        distances = np.linalg.norm(endpts_v1[0::2] - endpts_v1_int[0::2], axis=1) + np.linalg.norm(
            endpts_v1[1::2] - endpts_v1_int[1::2], axis=1)
        # Distances all data set (endpoints to intersections)
        distances_full = np.linalg.norm(endpts_v1_full[0::2] - endpts_v1_int_full[0::2], axis=1) + np.linalg.norm(
            endpts_v1_full[1::2] - endpts_v1_int_full[1::2], axis=1)

        return distances, distances_full, num_inl, l_matches_inliers, ind_l_matches_inliers

    ##IF IT IS USED THE TRIPLET MATCHES

    def get_error3(self, model):
        """
        NOT USED
        As there are few 3 view matches, algorithms do not work
        
        # This is another way to find error verifying epilolar lines and endpoints.
        # Here it is just evaluated correspondences in the 3 views and counting inliers
        # when it met the criteria three times 1)views 0-1, 2) views 1-2, views 2-0
        # (or twice maybe to be softer)

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.

        Returns
        -------
        error : TYPE
            DESCRIPTION.
        error_inl : TYPE
            DESCRIPTION.
        num_inl : TYPE
            DESCRIPTION.

        """



        P0, P1, P2 = model[0], model[1], model[2]
        # P0, P1, P2 = ransac_fit[0], ransac_fit[1], ransac_fit[2]

        # Finding fundamental matrices
        rR0, rt0 = relativeCameraMotion(P0, P1)
        F0 = F_from_KRt2(self.K, rR0, rt0)
        rR1, rt1 = relativeCameraMotion(P1, P2)
        F1 = F_from_KRt2(self.K, rR1, rt1)
        rR2, rt2 = relativeCameraMotion(P2, P0)  # could be the other way around (P0,P2)
        F2 = F_from_KRt2(self.K, rR2, rt2)

        # triplet correspondances
        tri_match = find_correspondences_3v(domain, self.view0, self.view1, self.view2, n_corr="full")
        # views 0-1
        distances0, distances_full0, num_inl0, l_matches_inliers0, ind_inl0 = self.find_scores_posesModel(self.domain,
                                                                                                          self.view0,
                                                                                                          self.view1,
                                                                                                          F=F0,
                                                                                                          tri_match=tri_match)
        # views 1-2
        distances1, distances_full1, num_inl1, l_matches_inliers1, ind_inl1 = self.find_scores_posesModel(self.domain,
                                                                                                          self.view1,
                                                                                                          self.view2,
                                                                                                          F=F1,
                                                                                                          tri_match=tri_match)
        # views 2-0
        distances2, distances_full2, num_inl2, l_matches_inliers2, ind_inl2 = self.find_scores_posesModel(self.domain,
                                                                                                          self.view2,
                                                                                                          self.view0,
                                                                                                          F=F2,
                                                                                                          tri_match=tri_match)

        # Checking inliers for the triplet
        l_matches_inliers0 = np.array(l_matches_inliers0)
        l_matches_inliers1 = np.array(l_matches_inliers1)
        l_matches_inliers2 = np.array(l_matches_inliers2)
        l_matches_inliers = []
        distances = 0
        for i, l0 in enumerate(l_matches_inliers0):
            if (l0[1] == l_matches_inliers1[:, 0]).sum():
                j = np.where((l0[1] == l_matches_inliers1[:, 0]).sum())[0][0]
                if (l_matches_inliers1[j][1] == l_matches_inliers2[:, 0]).sum():
                    k = np.where(l_matches_inliers1[j][1] == l_matches_inliers2[:, 0])[0][0]
                    l_matches_inliers.append(tri_match[i])
                    distances += distances_full0[i] + distances_full1[j] + distances_full2[k]

                    # print(distances0, distances1)
        error_inl = (distances) / (3 * len(l_matches_inliers))
        error = (distances) / (3 * len(l_matches_inliers))
        num_inl = len(l_matches_inliers)

        return error, error_inl, num_inl