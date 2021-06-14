import numpy as np

class Point:
    def __init__(self):
        self.id = 0
        self.coord = []
        self.desc = []
        self.view_id = 0
        self.angle = 0
        self.p = np.empty(3)
        self.p_hat = np.empty(3)
        self.coord_mod = [] #coordinate modified by epipolar constrain

class Point3D:
    def __init__(self, p_id, coord):
        self.id = p_id
        self.coord = coord