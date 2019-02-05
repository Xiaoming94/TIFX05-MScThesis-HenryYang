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

def resize_image(img):
    """
    Resizes an image to fit in a 20x20 box
    img -- a numpy array of the image
    """
    (h, w) = img.shape
    rf = 20/max(h, w)
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

def center_box_image(img):
    cm_x, cm_y = get_center_coord(img)
    centered_img = center_image_small(img,cm_x,cm_y)
    img_box = np.zeros([28,28])
    for i in range(20):
        for j in range(20):
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

def load_images(img_dir_path):
    images = os.listdir(img_dir_path)
    img_vec_list = []
    labels = []
    for img in images:
        img_arr = cv2.imread(os.path.join(img_dir_path,img), 0)
        img_arr = cv2.bitwise_not(img_arr)
        img_arr = unpad_img(img_arr)
        img_arr = resize_image(img_arr)
        img_processed = center_box_image(img_arr)
        img_vec_list.append(img_processed.flatten())
        labels.append(int(img[0]))
    
    return img_vec_list, labels