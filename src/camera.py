#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 20:01:06 2021

@author: pantoja
"""

"""
This codes belong in its original version to 
"Programming Computer Vision with Python" Jan Erik Solem
It follows the License 
Creative Commons Attribution-Noncommercial-No Derivative Works 3.0 United
States License.
http://creativecommons.org/licenses/by-nc-nd/3.0/us/
Small modifications performed
"""

import numpy as np
from scipy import linalg



class Camera(object):
    """ Class for representing pin-hole cameras. """
    
    def __init__(self,P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center
        
    
    def project(self,X):
        """    Project points in X (4*n array) and normalize coordinates. """
        
        x = np.dot(self.P,X)
        for i in range(3):
            x[i] /= x[2]    
        return x
        
        
    def factor(self):
        """    Factorize the camera matrix into K,R,t as P = K[R|t]. """
        
        # factor first 3*3 part
        K, R = linalg.rq(self.P[:,:3])
        
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1
        
        self.K = np.dot(K,T)
        self.R = np.dot(T,R) # T is its own inverse
        self.t = np.dot(linalg.inv(self.K),self.P[:,3])
        
        return self.K, self.R, self.t
        
    
    def center(self):
        """    Compute and return the camera center. """
    
        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -np.dot(self.R.T,self.t)
            return self.c

# helper functions    

def rotation_matrix(a):
    """    Creates a 3D rotation matrix for rotation
        around the axis of the vector a. """
    R = np.eye(4)
    R[:3,:3] = np.linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R
    

def rq(A):
   
    Q,R = linalg.qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T
    
    return R[:,::-1],Q[::-1,:]