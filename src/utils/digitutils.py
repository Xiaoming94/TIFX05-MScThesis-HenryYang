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
import threading
import ctypes

lwroutines = ctypes.CDLL("cout/liblw.so")
lwroutines.change_linewidth.argtypes = (ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int)
lwroutines.change_linewidth.restype = ctypes.POINTER(ctypes.c_int)

def resize_image(img, side):
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

def center_box_image(img, side, padding):
    cm_x, cm_y = get_center_coord(img)
    centered_img = center_image_small(img,cm_x,cm_y,side)
    img_box = np.zeros([side + (2 * padding),side + (2 * padding)])
    for i in range(side):
        for j in range(side):
            img_box[i+padding,j+padding] = centered_img[i,j]

    return img_box

def center_image_small(img,cm_x,cm_y,side):
    shift_x = int((side/2) - cm_x)
    shift_y = int((side/2) - cm_y)
    img_box = np.zeros([side,side])
    (h,w) = img.shape
    for i in range(h):
        for j in range(w):
            img_box[(i + shift_y) % side, (j + shift_x) % side] = img[i,j]

    return img_box

def unpad_img(img):
    img = img[~np.all(img == 0,axis = 1)]
    idx = np.argwhere(np.all(img[..., :] == 0, axis = 0))
    img = np.delete(img, idx, axis=1)
    return img

def change_linewidth(img, radius):
    global lwroutines
    h,w = img.shape
    img_array = img.astype('int').flatten().tolist()
    arr_type = ctypes.c_int * (h * w)
    new_img_arr = lwroutines.change_linewidth(arr_type(*img_array),ctypes.c_int(w),ctypes.c_int(h),ctypes.c_int(radius))
    l_new_img = []
    for i in range(h*w):
        l_new_img.append(new_img_arr[i])
    
    new_img = np.array(l_new_img).reshape(h,w).astype('float')
    #def find_new_pixval(img, newimg, x, y, radius,search_coords):
    #    (height,width) = img.shape
    #    pixels = set()
    #    for rx,ry in search_coords:
    #        # Bounding the coordinates
    #        drx = max(x + rx, 0)
    #        drx = min(drx, width-1)
    #        dry = max(y + ry, 0)
    #        dry = min(dry, height-1)
#
    #        pixels.add(img[dry,drx])
    #    
    #    newimg[y,x] = max(pixels) if radius > 0 else min(pixels)
#
    #(height,width) = img.shape
    #new_img = np.zeros([height,width])
    #a_radius = abs(radius)
    #l_rx = list(range(-1 * a_radius, a_radius + 1))
    #l_ry = list(range(-1 * a_radius, a_radius + 1))
    #search_coords = [(rx,ry) for rx in l_rx for ry in l_ry if (abs(rx) + abs(ry) <= a_radius)]
#
    #threads = []
    #for i in range(height):
    #    for j in range(width):
    #        #t = threading.Thread(target = find_new_pixval, args = (img,new_img,j,i,radius,search_coords))
    #        #threads.append(t)
    #        #t.start()
    #        find_new_pixval(img,new_img,j,i, radius, search_coords)
    #
    ##for t in threads:
    ##    t.join()
    
    return new_img
def intern_calc_linewidth(img):
    thickened = change_linewidth(img,1)
    thinned = change_linewidth(img,-1)
    gradient = thickened - thinned
    sumthick = np.sum(img)
    sumgrad = np.sum(gradient)
    return 2 * sumthick/sumgrad

def calc_linewidth(imgs):

    tau = 0
    with Pool(6) as p:
        thicknesses = np.array(p.map(intern_calc_linewidth,imgs))
        tau = np.mean(thicknesses)
    return tau

def load_image(img_path,bw):
    img = cv2.imread(img_path, 0)
    img = cv2.bitwise_not(img)
    if bw:
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img

def load_image_data(img_dir_path, side=20, padding=4, unpad=True, bw = False):
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
    img_list, labels = load_images(img_dir_path,bw)

    if (unpad):
        img_list = list(map(unpad_img, img_list))
    
    img_list = list(map(
        lambda img: center_box_image(resize_image(img, side), side, padding)
    ,img_list))
    return np.array(img_list), labels

def load_images(img_dir_path,bw):
    images = os.listdir(img_dir_path)
    img_list = []
    labels = []
    for img in images:
        img_list.append(load_image(os.path.join(img_dir_path,img),bw))
        labels.append(int(img[0]))
    
    return img_list, np.array(labels, dtype=np.int)