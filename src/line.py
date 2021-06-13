#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:24:13 2021

@author: pantoja
"""

import numpy as np

class Line:
    def __init__(self):
        self.id = -1
        self.local_id = -1
        self.points = []
        self.angle = []
        #self.angle2 = []
        #self.desc = []
        self.view_id = -1
        self.n = np.empty(3)
    
    def comp_angle(self):
        """
        Description
        -------
        Computes angle of a 2D line

        Returns
        -------
        None.

        """
        #print(self.points)
        #print(np.array(self.points[0].coord))
        #print(np.array(self.points[1].coord))
        
        smooth = 1e-7
        mid_point = 0.5*(np.array(self.points[0].coord) + np.array(self.points[1].coord))
        vect = np.array(self.points[0].coord) - mid_point
        ang = np.arctan((vect[1]+smooth)/(vect[0]+smooth))
        #vect2 = np.array(self.points[1].coord) - mid_point
        #ang2 = np.arctan(vect2[1]+smooth/vect2[0]+smooth)
        
        
        self.angle = ang
        #self.angle2 = ang2
        self.points[0].angle = ang
        self.points[1].angle = ang

class Line3D:
    def __init__(self, l_id, points):
        self.id = l_id
        self.points = points
