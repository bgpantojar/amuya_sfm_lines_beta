#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:14:38 2021
@author: pantoja
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from vp_detection import VPDetection
import os
import pandas as pd

class Domain:
    """

    Description
    -------
    This class contains all the informations related with the structure and 
    motion. Needs initial features to be created and use them in functions
    necessary to execute sfm pipeline.

    """    
    
    
    def __init__(self, points2d, lines2d, views, K):
        """
        
        Description
        -------
        Initialize a object from the class Domain

        Parameters
        ----------
        points2d : list
            List with objects from the class Point correspondent to the line
            segments endpoints.
        lines2d : list
            List with objects from the class Line correspondent to the line 
            segments detected from lsd.
        views : List with objects from the class view correspondent to each input
            image.
        K : numpy.ndarray
            Matrix with camera intrinsics.

        Returns
        -------
        None.
        

        """
        self.points2d = points2d
        self.lines2d = lines2d
        self.points3d = []
        self.lines3d = []
        self.views = views
        self.line_match = {}
        self.kps_match = []
        self.F_matrices = []
        self.P_rel = []
        self.K = K
        self.tracker = None

    def match_lines(self, plot=False, triplet=True):
        """
        
        Description
        -------
        This function will fill line matches between image pairs following
        the sequence how user input the image
        

        Parameters
        ----------
        plot : bool, optional
            If True, plots with line correspondences are done. The default is False.
        triplet : bool, optional
            If True, the line correspondences are found between lines i and i+2. The default is True.

        Returns
        -------
        None.        


        """

        print("SIFT descriptor based line matching-------------")

        self.line_match = {}  # clearing variable
        
        for view_id in range(len(self.views) - 1):
            # for view_id in range(len(domain.views)-1):

            # Lines are matched if ratio dist1/dist2 < 0.75

            # Reading descriptors
            desc1 = np.array(self.views[view_id].desc_lines)
            desc2 = np.array(self.views[view_id + 1].desc_lines)
            
            # Normalizing descriptors
            desc1 = desc1 / np.linalg.norm(desc1)
            desc2 = desc2 / np.linalg.norm(desc2)

            # Dividing descriptors into two groups. Initial (even) and Final (odd) endpoints
            desc1_ini = desc1[0::2]
            desc1_fin = desc1[1::2]
            desc2_ini = desc2[0::2]
            desc2_fin = desc2[1::2]

            # Findig euclidean distances (cdist scipy) for four cases of the two scenaries of matching.
            # Scenary A -> initial endpoint line image 1 match with initial endpoint image 2 (cases 1 and 2)
            # Scenary B -> initial endpoint line image 1 match with final endpoint image 2 (cases 3 and 4)
            # Cases: 1)desc1_ini&desc2_ini 2)desc1_fin&desc2_fin
            # Cases: 3)desc1_ini&desc2_ini 4)desc1_ini&desc2_ini

            dist_ini1_ini2 = cdist(desc1_ini, desc2_ini)
            dist_fin1_fin2 = cdist(desc1_fin, desc2_fin)

            dist_ini1_fin2 = cdist(desc1_ini, desc2_fin)
            dist_fin1_ini2 = cdist(desc1_fin, desc2_ini)

            # Line distances for two scenaries
            # Scenary A
            line_distA = dist_ini1_ini2 + dist_fin1_fin2
            # Scenary B
            line_distB = dist_ini1_fin2 + dist_fin1_ini2

            # Array with line distances taking the best option (ini-ini) or (ini-fin)
            line_dist = (line_distA <= line_distB) * line_distA + (line_distB < line_distA) * line_distB

            # Selecting the two minimum distances (two most similar lines to each line)
            args_min_2dist = np.argsort(line_dist)[:, 0:2]

            # Verifying if meet with distance condition mindist<0.8*2ndmindist
            # If so, create match

            matches = []
            for ind1, inds2 in enumerate(args_min_2dist):
                if line_dist[ind1, inds2[0]] < 0.75 * line_dist[ind1, inds2[1]]:
                    matches.append([ind1, inds2[0]])
                    # print("meet", matches)

            matches_ids = []  # list with matches following lines ids
            for inds in matches:
                id_line_v1 = self.views[view_id].lines[inds[0]].id
                id_line_v2 = self.views[view_id + 1].lines[inds[1]].id
                matches_ids.append([id_line_v1, id_line_v2])

            # Updating line_match_list (first element views ids tuple, second element matches list)
            tup_views = (view_id, view_id + 1)
            self.line_match[str(view_id) + str(view_id + 1)] = [tup_views, matches_ids]

            if plot:

                print("   Ploting line correspondences views: ", view_id, " and ", view_id + 1)

                # Reading images
                im1 = self.views[view_id].image
                im2 = self.views[view_id + 1].image

                # Creating side by side image
                sh1 = im1.shape
                sh2 = im2.shape
                shift_x = sh1[1]
                im = np.zeros((max(sh1[0], sh2[0]), sh1[1] + sh2[1], 3))
                im[0:sh1[0], 0:sh1[1], :] = im1
                im[0:sh2[0], sh1[1]:, :] = im2
                im = im.astype('uint8')
                
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # Ploting
                plt.figure()
                for l in matches_ids:
                    c = np.random.rand(3, )
                    
                    # line view i
                    x = [self.lines2d[l[0]].points[0].coord[0], self.lines2d[l[0]].points[1].coord[0]]
                    y = [self.lines2d[l[0]].points[0].coord[1], self.lines2d[l[0]].points[1].coord[1]]
                    plt.plot(x, y, color=c, linewidth=3)
                    # plt.annotate(str(self.lines2d[l[0]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")
                    
                    # line view i+1
                    x = [self.lines2d[l[1]].points[0].coord[0] + shift_x,
                         self.lines2d[l[1]].points[1].coord[0] + shift_x]
                    y = [self.lines2d[l[1]].points[0].coord[1], self.lines2d[l[1]].points[1].coord[1]]
                    plt.plot(x, y, color=c, linewidth=3)
                    # plt.annotate(str(self.lines2d[l[1]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")
                
                plt.imshow(im, cmap='gray')
                plt.savefig("../../output/" + self.views[view_id].name[:-4] + "_" + self.views[view_id + 1].name[
                                                                                    :-4] + "_matcher.png", dpi=1000)

            # This part will do the match between initial view and the third in each iteration
            # This would need to be rethought. It is used to calculate triplet scores for poses_ransac
            if view_id < len(self.views) - 2 and triplet:
                # Reading descriptors
                desc3 = np.array(self.views[view_id + 2].desc_lines)

                # Normalizing descriptors
                desc3 = desc3 / np.linalg.norm(desc3)

                # Dividing descriptors into two groups. Initial (even) and Final (odd) endpoints
                desc3_ini = desc3[0::2]
                desc3_fin = desc3[1::2]

                # Findig euclidean distances (cdist scipy) for four cases of the two scenaries of matching.

                dist_ini1_ini3 = cdist(desc1_ini, desc3_ini)
                dist_fin1_fin3 = cdist(desc1_fin, desc3_fin)

                dist_ini1_fin3 = cdist(desc1_ini, desc3_fin)
                dist_fin1_ini3 = cdist(desc1_fin, desc3_ini)

                # Line distances for two scenaries
                # Scenary A
                line_distA = dist_ini1_ini3 + dist_fin1_fin3
                # Scenary B
                line_distB = dist_ini1_fin3 + dist_fin1_ini3

                # Array with line distances taking the best option (ini-ini) or (ini-fin)
                line_dist = (line_distA <= line_distB) * line_distA + (line_distB < line_distA) * line_distB

                # Selecting the two minimum distances (two most similar lines to each line)
                args_min_2dist = np.argsort(line_dist)[:, 0:2]

                # Verifying if meet with distance condition mindist<0.8*2ndmindist
                # If so, create match

                matches = []
                for ind1, inds3 in enumerate(args_min_2dist):
                    if line_dist[ind1, inds3[0]] < 0.75 * line_dist[ind1, inds3[1]]:
                        matches.append([ind1, inds3[0]])
                        # print("meet", matches)

                matches_ids = []  # list with matches following lines ids
                for inds in matches:
                    id_line_v1 = self.views[view_id].lines[inds[0]].id
                    id_line_v3 = self.views[view_id + 2].lines[inds[1]].id
                    matches_ids.append([id_line_v1, id_line_v3])

                # Updating line_match_list (first element views ids tuple, second element matches list)
                tup_views = (view_id, view_id + 2)
                self.line_match[str(view_id) + str(view_id + 2)] = [tup_views, matches_ids]

    def match_lines_manual(self, plot=False, triplet=True):
        """
        
        Description
        -------
        This function will fill line matches between image pairs following
        the sequence how user input the image. This follows manual annotations
        
        Parameters
        ----------
        plot : bool, optional
            If True, plots with line correspondences are done. The default is False.
        triplet : bool, optional
            If True, the line correspondences are found between lines i and i+2. The default is True.

        Returns
        -------
        None.
       

        """
        
        print("SIFT descriptor based line matching-------------")

        self.line_match = {}  # clearing variable
        j = 0
        for view_id in range(len(self.views) - 1):
            df = pd.read_csv(os.path.join('../../manual_line_annotation', self.views[0].name[:-4] + '_csv.csv'))
            nb_lines = df.shape[0]
            matches_ids = [[i + j, i + j + nb_lines] for i in range(nb_lines)]
            tup_views = (view_id, view_id + 1)
            
            self.line_match[str(view_id) + str(view_id + 1)] = [tup_views, matches_ids]
            j += nb_lines

            if plot:

                print(" Ploting line correspondences views: ", view_id, " and ", view_id + 1)

                # Reading images
                im1 = self.views[view_id].image
                im2 = self.views[view_id + 1].image
                
                # Creating side by side image
                sh1 = im1.shape
                sh2 = im2.shape
                shift_x = sh1[1]
                im = np.zeros((max(sh1[0], sh2[0]), sh1[1] + sh2[1], 3))
                im[0:sh1[0], 0:sh1[1], :] = im1
                im[0:sh2[0], sh1[1]:, :] = im2
                im = im.astype('uint8')
                
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # Ploting
                plt.figure()
                for l in matches_ids:
                    c = np.random.rand(3, )
                    # line view i
                    x = [self.lines2d[l[0]].points[0].coord[0], self.lines2d[l[0]].points[1].coord[0]]
                    y = [self.lines2d[l[0]].points[0].coord[1], self.lines2d[l[0]].points[1].coord[1]]
                    plt.plot(x, y, color=c, linewidth=3)
                    # plt.annotate(str(self.lines2d[l[0]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")
                    
                    # line view i+1
                    x = [self.lines2d[l[1]].points[0].coord[0] + shift_x,
                         self.lines2d[l[1]].points[1].coord[0] + shift_x]
                    y = [self.lines2d[l[1]].points[0].coord[1], self.lines2d[l[1]].points[1].coord[1]]
                    plt.plot(x, y, color=c, linewidth=3)
                    # plt.annotate(str(self.lines2d[l[1]].id), ((x[0]+x[1])/2,(y[0]+y[1])/2), ((x[0]+x[1])/2,(y[0]+y[1])/2), c="red")
                    
                plt.imshow(im, cmap='gray')
                plt.savefig("../../output/" + self.views[view_id].name[:-4] + "_" + self.views[view_id + 1].name[
                                                                                    :-4] + "_matcher.png", dpi=1000)
            if view_id == 0 and triplet:
                tup_views = (view_id, view_id + 2)
                matches_ids = [[i, i + 2 * nb_lines] for i in range(nb_lines)]
                self.line_match[str(view_id) + str(view_id + 2)] = [tup_views, matches_ids]

    def spherical_representation(self):
        """
        
        Description
        -------        
        This function will find the unit-spherical representation of lines and points for all the views

        Returns
        -------
        None.

        """

        for view in self.views:
            
            im_shape = view.image.shape
            cam_cent_coord = np.array([im_shape[1] / 2, im_shape[
                0] / 2])  # The vectors p and n for points and lines are based on the camara center (assumed at the middle of the image plane)
            
            for line in view.lines:
                for pt in line.points:
                    
                    p = np.concatenate((np.array(pt.coord) - cam_cent_coord, [self.K[0, 0]]))
                    pt.p = p / (np.linalg.norm(p))
                    
                n = np.cross(line.points[0].p, line.points[1].p)
                line.n = n / np.linalg.norm(n)

    def plot_lines(self, lines_list, view_id, name_plot):
        """
        Description
        -----
        Given a list of line objects correspondences, it plots the lines in
        views i and i+1
        

        Parameters
        ----------
        lines_list : list
            list of line objects correspondences.
        view_id : int
            id view i.
        name_plot : str
            name of plot to save file.

        Returns
        -------
        None.

        """

        # Reading images
        im1 = self.views[view_id].image
        im2 = self.views[view_id + 1].image

        # Creating side by side image
        sh1 = im1.shape
        sh2 = im2.shape
        shift_x = sh1[1]
        im = np.zeros((max(sh1[0], sh2[0]), sh1[1] + sh2[1], 3))
        im[0:sh1[0], 0:sh1[1], :] = im1
        im[0:sh2[0], sh1[1]:, :] = im2
        im = im.astype('uint8')
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Ploting
        plt.figure()
        for l in lines_list:
            c = np.random.rand(3, )
            # line view i
            x = [self.lines2d[l[0]].points[0].coord[0], self.lines2d[l[0]].points[1].coord[0]]
            y = [self.lines2d[l[0]].points[0].coord[1], self.lines2d[l[0]].points[1].coord[1]]
            # x = [domain.lines2d[l[0]].points[0].coord[0], domain.lines2d[l[0]].points[1].coord[0]]
            # y = [domain.lines2d[l[0]].points[0].coord[1], domain.lines2d[l[0]].points[1].coord[1]]
            # print(x,y)
            plt.plot(x, y, color=c, linewidth=3)
            plt.annotate(str(self.lines2d[l[0]].id), ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2),
                         ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2), c="red")
            # line view i+1
            x = [self.lines2d[l[1]].points[0].coord[0] + shift_x, self.lines2d[l[1]].points[1].coord[0] + shift_x]
            y = [self.lines2d[l[1]].points[0].coord[1], self.lines2d[l[1]].points[1].coord[1]]
            # x = [domain.lines2d[l[1]].points[0].coord[0]+shift_x, domain.lines2d[l[1]].points[1].coord[0]+shift_x]
            # y = [domain.lines2d[l[1]].points[0].coord[1], domain.lines2d[l[1]].points[1].coord[1]]
            # print(x,y)
            plt.plot(x, y, color=c, linewidth=3)
            plt.annotate(str(self.lines2d[l[1]].id), ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2),
                         ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2), c="red")
        plt.imshow(im, cmap='gray')
        plt.savefig("../../output/" + self.views[view_id].name[:-4] + "_" + self.views[view_id + 1].name[
                                                                            :-4] + "_" + name_plot + ".png", dpi=1000)

    def match_kps(self, plot=False):
        """
        
        Description
        -------
        This function will fill sift keypoint matches between image pairs following
        the sequence how user input the image

        Parameters
        ----------
        plot : bool, optional
            If true, sift keycorrespondances are plot. The default is False.

        Returns
        -------
        None.        

        

        """

        

        print("SIFT descriptor kps matching-------------")

        self.kps_match = []  # clearing variable
        
        for view_id in range(len(self.views) - 1):

            # Reading kps

            kp1 = np.array(self.views[view_id].kps)
            kp2 = np.array(self.views[view_id + 1].kps)

            # Reading descriptors
            desc1 = np.array(self.views[view_id].desc_kps)
            desc2 = np.array(self.views[view_id + 1].desc_kps)

            # Normalizing descriptors
            desc1 = desc1 / np.linalg.norm(desc1)
            desc2 = desc2 / np.linalg.norm(desc2)

            # print(len(kp1))

            # Matching using sift criteria
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.75 * n.distance:
                    good.append(m)
                    matchesMask[i] = [1, 0]

            if plot:
                print("Ploting kps correspondences views: ", view_id, " and ", view_id + 1)

                # params to draw kps correspondences
                draw_params = dict(matchColor=(0, 255, 0),
                                   singlePointColor=(255, 0, 0),
                                   matchesMask=matchesMask,
                                   flags=cv2.DrawMatchesFlags_DEFAULT)

                im1 = cv2.cvtColor(self.views[view_id].image, cv2.COLOR_RGB2GRAY)
                im2 = cv2.cvtColor(self.views[view_id + 1].image, cv2.COLOR_RGB2GRAY)
                img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, matches, None, **draw_params)
                plt.figure()
                plt.imshow(img3), plt.show()
                plt.savefig("../../output/" + self.views[view_id].name[:-4] + "_" + self.views[view_id + 1].name[
                                                                                    :-4] + "_matcher_kps.png", dpi=1000)
            
            if len(good) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


            # Updating kps_match_list (views ids tuple | good matches | souce points (viewi) | destini points (viewi+1)
            tup_views = (view_id, view_id + 1)
            self.kps_match.append(
                [tup_views, good, src_pts.reshape((len(src_pts), 2)), dst_pts.reshape((len(dst_pts), 2))])

    def find_fundamental_matrices(self):
        """
        Description
        ------
        No corresponds to line sfm pipeline
        Find fundamental matrices for each pair of views sequentially

        Returns
        -------
        None.

        """
        # Find F with opencv
        # See https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

        print("Funtamental Matrices Computation - Epipolar Geometry-------------")

        self.F_matrices = []

        for view_id in range(len(self.views) - 1):
            src_pts = self.kps_match[view_id][2]
            dst_pts = self.kps_match[view_id][3]

            # F: fundamental matrix. Mask: binary array indicating inlier points in
            # F computation
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS, 10)

            tup_views = (view_id, view_id + 1)
            self.F_matrices.append([tup_views, F, mask])

    def find_relative_poses_from_E(self, K):
        """
        Description
        --------
        No corresponds to line sfm pipeline
        Find relative poses using essential matrix calculated with sift keypoints

        Parameters
        ----------
        K : numpy.ndarray
            Matrix with intrinsic parameters of the camera.

        Returns
        -------
        None.

        """

        print("Camera Projection Matrices from E -------------")

        self.P_rel = []

        for view_id in range(len(self.views) - 1):
            src_pts = self.kps_match[view_id][2]
            dst_pts = self.kps_match[view_id][3]

            # F: fundamental matrix. Mask: binary array indicating inlier points in
            # F computation
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC)
            inliers = np.where(mask)[0]  # inliers in E estimation
            x1_inl = src_pts[inliers, :]
            x2_inl = dst_pts[inliers, :]
            # x1 = src_pts.T
            # x2 = dst_pts.T
            P0 = K @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
            _, R2, t2, maskP = cv2.recoverPose(E, x1_inl, x2_inl, K)
            inliers_P = np.where(maskP)[0]  # inliers in pose recovering
            x1_inl = x1_inl[inliers_P, :].T
            x2_inl = x2_inl[inliers_P, :].T
            P1 = K @ np.concatenate((R2, t2), axis=1)

            tup_views = (view_id, view_id + 1)
            self.P_rel.append([tup_views, P0, P1, E, mask, x1_inl, x2_inl])

    def cluster_lines(self, K, save_flag):
        """
        
        Description
        -------
        Cluster group of parallel lines following Manhattan world representation

        Parameters
        ----------
        K : numpy.ndarray
            Matrix with intrinsic parameters of the camera.
        save_flag : bool
            If true, images with the lines clulstered are saved.

        Returns
        -------
        None.

        """
        id = 0
        for view_id in range(len(self.views)):
            nb_lines = len(self.views[view_id].lines)
            all_lines_coords = np.zeros((nb_lines, 4))  # each row contains x1, y1, x2, y2
            for line_id_local in range(nb_lines):
                all_lines_coords[line_id_local, 0] = self.views[view_id].lines[line_id_local].points[0].coord[0]
                all_lines_coords[line_id_local, 1] = self.views[view_id].lines[line_id_local].points[0].coord[1]
                all_lines_coords[line_id_local, 2] = self.views[view_id].lines[line_id_local].points[1].coord[0]
                all_lines_coords[line_id_local, 3] = self.views[view_id].lines[line_id_local].points[1].coord[1]

            # vp detection
            length_thresh = 0  # Minimum length of the line in pixels
            principal_point = (K[0, -1], K[1, -1])  # Specify a list or tuple of two coordinates
            
            # First value is the x or column coordinate
            # Second value is the y or row coordinate
            focal_length = (K[0, 0] + K[1, 1]) / 2  # Specify focal length in pixels
            seed = 0  # Or specify whatever ID you want (integer)
            vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
            _ = vpd.find_vps(self.views[view_id].image, all_lines_coords)
            if save_flag:
                save_path = os.path.join('../../output/', self.views[view_id].name)
            else:
                save_path = ''
            vpd.create_debug_VP_image(show_image=False, save_image=save_path)  # save_image=save_image)
            sorted_cluster_ids = np.argsort([len(cluster_id) for cluster_id in vpd.clusters])[::-1]
            self.views[view_id].parallel_line_ids.append(
                vpd.clusters[sorted_cluster_ids[0]] + id)  # first dominant parallel lines ids
            self.views[view_id].parallel_line_ids.append(
                vpd.clusters[sorted_cluster_ids[1]] + id)  # second dominant parallel lines ids
            id = id + nb_lines

    def plot_lines3d(self, color='k'):
        """
        Description
        ---------
        Ploting 3D lines triangulated and saved in domain.lines3d.

        Parameters
        ----------
        color : str, optional
            Lines color. The default is 'k'.

        Returns
        -------
        None.

        """
        
        x_coords = []
        y_coords = []
        z_coords = []
        for l in self.lines3d:
            x_coords.append(l.points[0].coord[0])
            x_coords.append(l.points[1].coord[0])
            y_coords.append(l.points[0].coord[1])
            y_coords.append(l.points[1].coord[1])
            z_coords.append(l.points[0].coord[2])
            z_coords.append(l.points[1].coord[2])
        q0 = np.quantile(x_coords, (0, 1))
        q1 = np.quantile(y_coords, (0, 1))
        q2 = np.quantile(z_coords, (0, 1))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        coords = []
        for l in self.lines3d:
            coords.append(l.points[0].coord)
            if (l.points[0].coord[0] > q0[0]) & (l.points[1].coord[0] > q0[0]):
                if (l.points[0].coord[0] < q0[1]) & (l.points[1].coord[0] < q0[1]):
                    if (l.points[0].coord[1] > q1[0]) & (l.points[1].coord[1] > q1[0]):
                        if (l.points[0].coord[1] < q1[1]) & (l.points[1].coord[1] < q1[1]):
                            if (l.points[0].coord[2] > q2[0]) & (l.points[1].coord[2] > q2[0]):
                                if (l.points[0].coord[2] < q2[1]) & (l.points[1].coord[2] < q2[1]):
                                    # ax.scatter(l.points[0].coord[0], l.points[0].coord[1], l.points[0].coord[2], s=5, c='b')
                                    # ax.scatter(l.points[1].coord[0], l.points[1].coord[1], l.points[1].coord[2], s=5, c='r')
                                    ax.plot([l.points[0].coord[0], l.points[1].coord[0]],
                                            [l.points[0].coord[1], l.points[1].coord[1]],
                                            [l.points[0].coord[2], l.points[1].coord[2]], c=color)

        plt.axis('on')