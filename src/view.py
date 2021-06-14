import numpy as np
import cv2

class View:
    def __init__(self):
        self.id = 0
        self.name = ""
        self.image = np.empty(0)
        self.points = []
        self.lines = []
        self.kps_lines = []
        self.desc_lines = []
        self.kps = []
        self.desc_kps = []
        self.parallel_line_ids = []
        self.P = []
        
    def find_line_desc(self):
        """
        Description
        ---------
        Computes line descriptors based on sift descriptors for endpoints

        Returns
        -------
        None.

        """        
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #img = cv.cvtColor(self.image, cv2.COLOR_BGR2RGB)
     
        #octave_index = 1
        keypoints = []
        for point in self.points:
            kp = cv2.KeyPoint()
            kp.pt = (point.coord[0],point.coord[1])
            kp.size = 8.0
            kp.angle = point.angle
            #kp.angle = .5
            kp.class_id = point.id
            keypoints.append(kp)
            
        sift = cv2.SIFT_create()
        self.kps_lines, self.desc_lines = sift.compute(gray, keypoints)
        
        #Assign descriptor to points
        for p, point in enumerate(self.points):
            point.desc = self.desc_lines[p]
            
    def find_kps(self):
        """
        Description
        --------
        #find kps and their descriptors following sift methodology

        Returns
        -------
        None.

        """

        
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
                         
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        self.kps, self.desc_kps = sift.detectAndCompute(gray,None)
