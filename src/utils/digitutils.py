'''
processimage.py

Author: Henry Yang (XiaoMing)

This is the file containing utility functions for processing handwritten digits into numerical numpy arrays
This file is assuming that the number is written on a white background only
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import functools
from multiprocessing import Pool

def resize_image(img, side=20):
    """
    Resizes an image to fit in a 20x20 box
    
    Parameter:
    img (np.array) : 

    Returns:
    return a numpy array that represents the resized image
    """
    (h, w) = img.shape
    rf = side/max(h, w)
    img_resized = cv2.resize(img,None,fx=rf,fy=rf,interpolation = cv2.INTER_CUBIC)
    return img_resized

def get_center_coord(img):
    intensity = np.sum(img)
    sum_x = 0
    sum_y = 0
    (h,w) = img.shape
    for i in range(h):
        for j  in range(w):
            sum_x += (j + 1) * img[i,j]
            sum_y += (i + 1) * img[i,j]

    cm_x = int(round(sum_x/intensity))
    cm_y = int(round(sum_y/intensity))

    return cm_x,cm_y

def center_box_image(img, side=20, padding=4):
    cm_x, cm_y = get_center_coord(img)
    centered_img = center_image_small(img,cm_x,cm_y)
    img_box = np.zeros([side + (2 * padding),side + (2 * padding)])
    for i in range(side):
        for j in range(side):
            img_box[i+4,j+4] = centered_img[i,j]

    return img_box

def center_image_small(img,cm_x,cm_y):
    shift_x = 10 - cm_x
    shift_y = 10 - cm_y
    img_box = np.zeros([20,20])
    (h,w) = img.shape
    for i in range(h):
        for j in range(w):
            img_box[(i + shift_y) % 20, (j + shift_x) % 20] = img[i,j]

    return img_box

def unpad_img(img):
    img = img[~np.all(img == 0,axis = 1)]
    idx = np.argwhere(np.all(img[..., :] == 0, axis = 0))
    img = np.delete(img, idx, axis=1)
    return img

def change_linewidth(img, radius):
    def find_new_pixval(img, x, y, radius):
        (height,width) = img.shape
        xstart = max(x - abs(radius), 0)
        xend = min(x + abs(radius), width)
        newpix = img[y,x]
        for rx in range(xstart,xend):
            dx = abs(rx - x) # Distance to X
            dy_search = abs(radius) - dx
            ystart = max(y - dy_search, 0)
            yend = min(y + dy_search, height)
            for ry in range(ystart,yend):
                if radius < 0:
                    newpix = min(img[ry,rx],newpix)
                else:
                    newpix = max(img[ry,rx],newpix)
            
        return newpix

    (height,width) = img.shape
    new_img = np.zeros([height,width])
    for i in range(height):
        for j in range(width):
            new_img[i,j] = find_new_pixval(img,j,i,radius) 
    
    return new_img
def intern_calc_linewidth(img):
    thickened = change_linewidth(img,1)
    thinned = change_linewidth(img,-1)
    gradient = thickened - thinned
    sumthick = np.sum(thickened)
    sumgrad = np.sum(gradient)
    return 2 * sumthick/sumgrad

def calc_linewidth(imgs):

    tau = 0
    with Pool(4) as p:
        thicknesses = np.array(p.map(intern_calc_linewidth,imgs))
        tau = np.mean(thicknesses)
    return tau

def load_image(img_path):
    img = cv2.imread(img_path, 0)
    img = cv2.bitwise_not(img)
    return img

def load_image_data(img_dir_path, side=20, padding=4, unpad=True):
    """
    Function that loades images of handwritten digits, applying preprocessing on them
    and finally returns them as numpy arrays.

    This function assumes that all images are placed in the directory specified by img_dir_path.
    It also assumes that the file names are named as following: <number>_<idx>.png.
    A third and final assumption is that the images have white background and black forground pixels

    It loads the images and applies the preprocessing steps as discribed on the MNIST website

    Parameter

    img_dir_path (String or Path): Path to the directory where the images are
    radius (integer): The linewidth change use for normalization. Set to 0 if nothing should be done about the images.
    unpad (boolean): tells the function weather or not to unpad the original images before applying preprocessing

    Returns:

    img_list(np.array), labels(np.array)

    The list of images as a 2d numpy array.
    Array of labels for the images
    """
    img_list, labels = load_images(img_dir_path)

    if (unpad):
        img_list = list(map(unpad_img, img_list))
    
    img_list = list(map(
        lambda img: center_box_image(resize_image(img, side), side, padding).flatten()
    ,img_list))
    return np.array(img_list), labels

def load_images(img_dir_path):
    images = os.listdir(img_dir_path)
    img_list = []
    labels = []
    for img in images:
        img_list.append(load_image(os.path.join(img_dir_path,img)))
        labels.append(int(img[0]))
    
    return img_list, np.array(labels, dtype=np.int)