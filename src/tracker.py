import numpy as np
from utils import *


class Tracker:
    def __init__(self):
        self.ids = []
        self.bin = []
        self.n_new_tracks = []

    def comp_tracker(self, domain, last_view=2):
        """
        Description
        --------
        # Compute the tracker for lines. Two tracker necessary. Arrayis with information
        # of the point 3D i in the view j. ids with the line ids. bi, if the line
        # i is seen in the view j.
        # initial=True if the tracker is performed for the 3 base views.
        # initial=False. Tracker will add points from new view (starting from view 3)
        # last_view: the tracker will take into account until this view

        Parameters
        ----------
        domain : Domain
            Domaing object with SfM information.
        last_view : int, optional
            last view for computation of tracker. The default is 2.

        Returns
        -------
        None.

        """



        # n_views = len(domain.views)
        # Tr_ij = np.empty(shape=[0,n_views]) #lines ids of track i in view j
        Tr_ij = np.empty(shape=[0, last_view + 1])  # lines ids of track i in view j
        # Tr_bi = np.empty(shape=[0,n_views]) #binary array marking if track is in view
        Tr_bi = np.empty(shape=[0, last_view + 1])  # binary array marking if track is in view
        n_new_tracks = []  # number of new tracks per new view added

        # for v in range(n_views-1):
        for v in range(last_view):
            # if initial==True:
            # matches = domain.line_match["{}{}".format(v,v+1)][1]
            P0 = domain.views[v].P
            P1 = domain.views[v + 1].P
            # Finding fundamental matrices
            rR, rt = relativeCameraMotion(P0, P1)
            F = F_from_KRt2(domain.K, rR, rt)

            # duplet correspondances that meet the criteria
            num_inl, l_matches_inliers = verify_duplet(domain, v, v + 1, F, 0.1, tri_match=None)

            if num_inl == 0:
                print("there are not lines that meet with epipolar criteria!")

            if v > 0:  # verify if line was already observed in the last iteration
                check_last_it = np.array([l in Tr_ij[:, v] for l in l_matches_inliers[:, 0]])

                # Lines already observed and new contribution by new registered view
                repeated_lines = l_matches_inliers[check_last_it]
                new_lines = l_matches_inliers[check_last_it == False]

                # Registering repeated lines in existent tracks
                for l in repeated_lines:
                    ind = np.where((l[0] == Tr_ij[:, v]))
                    Tr_ij[ind, v + 1] = l[1]
                    Tr_bi[ind, v + 1] = 1

                # creating new tracks
                # Tr_iteration = np.zeros((len(new_lines), n_views), dtype='int')
                Tr_iteration = np.zeros((len(new_lines), last_view + 1), dtype='int')
                # Tr_iteration_bi = np.zeros((len(new_lines), n_views), dtype='int')
                Tr_iteration_bi = np.zeros((len(new_lines), last_view + 1), dtype='int')
                Tr_iteration[:, v:v + 2] = np.copy(new_lines)
                Tr_iteration_bi[:, v:v + 2] = np.ones_like(new_lines)
                n_new_tracks.append(len(new_lines))

            elif v == 0:
                # Tr_iteration = np.zeros((len(l_matches_inliers), n_views), dtype='int')
                Tr_iteration = np.zeros((len(l_matches_inliers), last_view + 1), dtype='int')
                # Tr_iteration_bi = np.zeros((len(l_matches_inliers), n_views), dtype='int')
                Tr_iteration_bi = np.zeros((len(l_matches_inliers), last_view + 1), dtype='int')
                Tr_iteration[:, v:v + 2] = np.copy(l_matches_inliers)
                Tr_iteration_bi[:, v:v + 2] = np.ones_like(l_matches_inliers)
                n_new_tracks.append(len(l_matches_inliers))

            Tr_ij = np.concatenate((Tr_ij, Tr_iteration))
            Tr_bi = np.concatenate((Tr_bi, Tr_iteration_bi))

        self.ids = Tr_ij
        self.bin = Tr_bi
        self.n_new_tracks = n_new_tracks
