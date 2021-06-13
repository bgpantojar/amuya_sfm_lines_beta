import cv2
import os
import glob
import numpy as np

def lsd(full_image_name, image_path, line_path):
    """
    Description
    -------
    This function calls lsd binary files from (http://www.ipol.im/pub/art/2012/gjmr-lsd/)
    to compute the line segments.

    Parameters
    ----------
    full_image_name : str
        The name + extension of the image file. Example: 'car.JPG'
    image_path : str
        The path of the image file.
    line_path : str
        The path to save the lsd output results.
    """
    
    #Check if line_path exist. If not, creat it
    check_dir = os.path.isdir(line_path)
    if not check_dir:
        os.makedirs(line_path)
    
    file_name, file_extension = os.path.splitext(full_image_name)

    # create and save the pgm file, which is a graysacle image file.
    img_c = cv2.imread(os.path.join(image_path, full_image_name))
    img_g = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
    im_n = os.path.join(line_path, file_name + '.pgm')
    cv2.imwrite(im_n, img_g)
    # define the output paths
    out_im = os.path.join(line_path, file_name + '.eps')
    out_txt = os.path.join(line_path, file_name + '.txt')
    # cmmd = str("lsd/lsd -P "+out_im+" -s .07 "+im_n+" "+out_txt) school
    os.system('chmod u+x lsd/lsd')
    cmmd = str("lsd/lsd -P " + out_im + " -s .1 " + im_n + " " + out_txt)
    os.system(cmmd)
    print('processed', file_name)
    
    #Reading lsd output and creating lines and points lists to be returned    
    txt_file = open(out_txt)
    lines = []
    points = []
    for x in txt_file:
        lines.append(list((np.float_(x[:-2].split(" ")))[:-3]))
        points.append(lines[-1][:2])
        points.append(lines[-1][2:4])
        
    return img_c, lines, points   



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    