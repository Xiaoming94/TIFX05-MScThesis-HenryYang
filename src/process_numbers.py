import utils
from utils import digitutils as dutils
import os
import numpy as np
import cv2
from console_progressbar import ProgressBar
from multiprocessing import Pool, Lock, Value
import time

mnist_linethickness = 67.14082725553595

min_thickness = 25
max_thickness = 80

class ParallellWidthAdjust(object):

    def __init__(self):
        self.lock = Lock()

    def reset(self,factor,pb):
        self.factor = factor
        self.iterations = Value('i',0)
        self.pb = pb

    def adjust_linewidth(self,img):
        new_img = utils.change_linewidth(img,self.factor)
        self.report_change()
        return new_img

    def report_change(self):
        with self.lock:
            self.iterations.value += 1
            self.pb.print_progress_bar(self.iterations.value)

pwa = ParallellWidthAdjust()


def scale_down(data):
    downscaled = list(map(
        lambda img: cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC), data
    ))

    return np.array(downscaled)

def wrapper_pwa_linewidth(img):
    return pwa.adjust_linewidth(img)

def change_thickness(data, factor):
    global pwa
    datapoints = data.shape[0]
    pb = ProgressBar(total=datapoints,prefix="processing digits:", length=20, fill="=",zfill="_")
    pwa.reset(factor,pb)
    with Pool(6) as p:
        new_data = p.map(wrapper_pwa_linewidth, data)
    return np.array(new_data)

def change_thickness_nosave(data, ref_thickness):
    print("Adjusting the thicknesses to %s" % ref_thickness)
    diff = 1
    r = 1
    curr_data = data
    tau = 0
    while abs(ref_thickness - tau) > diff:
        
        tau = utils.calc_linewidth(curr_data)
        print("current linethickness : %s" % tau)
        dt = ref_thickness - tau
        r = int(round((dt) * 0.25))
        r = r if r != 0 else int(np.sign(dt)) # Ensures that r is at least 1 or -1
        print("Adjusting Linewidth with r = %s" % r)
        if abs(ref_thickness - tau) > diff:
            curr_data = change_thickness(curr_data,r)
    return curr_data


def build_thickness_data(data, ref_thickness):
    tau_data_dict = {}
    diff = 1
    r = 1
    curr_data = data
    tau = 0
    print("Preparing testing data")
    while abs(ref_thickness - tau) > diff:
        
        tau = utils.calc_linewidth(curr_data)
        print("current linethickness : %s" % tau)
        tau_data_dict[tau] = scale_down(curr_data)
        r = int(np.sign(ref_thickness - tau))
        if abs(ref_thickness - tau) > diff:
            curr_data = change_thickness(curr_data,r)

    return tau_data_dict

def helper_normalize_digit(digit, tau):


def normalize_digit_thickness(digits):
    tau = utils.calc_linewidth(digits)


def load_process_digits(pathlist):
    digits = []
    labels = np.array([])
    for p in pathlist:
        digits_data, labels_data = utils.load_image_data(p,side=286, padding=57)
        labels = np.concatenate((labels,labels_data))
        digits_data = change_thickness_nosave(digits_data,min_thickness)
        digits.append(digits_data)
        print(labels.shape)
    d_combined = np.concatenate(digits,axis=0)
    print(d_combined.shape)
    return d_combined, labels

images_path = os.path.join(".","images")

xm_digits_path = os.path.join(images_path,"XiaoMing_Digits")
ob_digits_path = os.path.join(images_path,"60 Images")
m_digits_path = os.path.join(images_path,"mnumbers")
hubben1_path = os.path.join(images_path,"Size1")
hubben2_path = os.path.join(images_path,"Size2")
hubben3_path = os.path.join(images_path,"Size3")
hubben4_path = os.path.join(images_path,"Size4")

digit_paths = [
      xm_digits_path
    , ob_digits_path
    , m_digits_path
    , hubben1_path
    , hubben2_path
    , hubben3_path
    , hubben4_path
]

print("===== LOADING DIGITS =====")

d_combined, labels = load_process_digits(digit_paths)

data_dict = build_thickness_data(d_combined,max_thickness)
data_dict['labels'] = labels
utils.save_processed_data(data_dict,'combined_testing_data')

print("Script done running")