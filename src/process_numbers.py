import utils
from utils import digitutils as dutils
import os
import numpy as np
import cv2

mnist_linethickness = 67.14082725553595

def scale_down(data):
    downscaled = list(map(
        lambda img: cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC), data
    ))

    return np.array(downscaled)


def change_thickness(data, factor):

    new_data = list(map(
        lambda img: dutils.change_linewidth(img, factor), data   
    ))
    return np.array(new_data)

def build_thickness_data(data, ref_thickness):
    tau_data_dict = {}
    diff = 2
    r = 1
    curr_data = data
    tau = 0
    while abs(ref_thickness - tau) > diff:
        
        tau = utils.calc_linewidth(curr_data)
        tau_data_dict[tau] = scale_down(curr_data)
        r = int(np.sign(ref_thickness - tau))
        
        if abs(ref_thickness - tau) > diff:
            curr_data = change_thickness(curr_data,r)

    return tau_data_dict

xm_digits_path = os.path.join(".","images","XiaoMing_Digits")
ob_digits_path = os.path.join(".","images","60 Images")

print("===== LOADING DIGITS =====")

xm_digits, xm_labels = utils.load_image_data(xm_digits_path,side=286, padding=57)
ob_digits, ob_labels = utils.load_image_data(ob_digits_path,side=286, padding=57)
combined_data = np.concatenate((xm_digits,ob_digits))

print("===== CREATING LINETHICKNESS BASED DATABASE =====")

data_xm = build_thickness_data(xm_digits,mnist_linethickness)
data_ob = build_thickness_data(ob_digits,mnist_linethickness)
data_comb = build_thickness_data(combined_data, mnist_linethickness)

print("===== SAVING DATA TO DISK =====")

utils.save_processed_data(data_xm, "xiaoming_digits")
utils.save_processed_data(data_ob, "Oleks_digits")
utils.save_processed_data(data_comb, "combined_digits")