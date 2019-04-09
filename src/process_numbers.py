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
        tau, thick, thin = utils.linewidth_calculations(img)
        self.report_change()
        return tau, thick, thin

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
        results_list = p.map(wrapper_pwa_linewidth, data)
    
    taus = np.zeros(datapoints)
    new_data = np.zeros(data.shape)
    for i in range(datapoints):
        tau, thick, thin = results_list[i]
        taus[i] = tau
        new_data[i] = thick if factor > 0 else thin        

    return np.mean(taus), new_data

def change_thickness_nosave(data, ref_thickness):
    print("Adjusting the thicknesses to %s" % ref_thickness)
    diff = 1
    done_looping = False
    curr_data = data
    r = -1
    while not done_looping:
        
        tau,curr_data = change_thickness(curr_data,r)
        print("current linethickness : %s" % tau)
        r = int(np.sign(ref_thickness - tau))
        done_looping = abs(ref_thickness - tau) < diff
        
    return curr_data


#def build_thickness_data(data, ref_thickness):
#    tau_data_dict = {}
#    diff = 1
#    r = 1
#    curr_data = data
#    tau = 0
#    while abs(ref_thickness - tau) > diff:
#        
#        tau = utils.calc_linewidth(curr_data)
#        tau_data_dict[tau] = scale_down(curr_data)
#        r = int(np.sign(ref_thickness - tau))
#        
#        if abs(ref_thickness - tau) > diff:
#            curr_data = change_thickness(curr_data,r)
#
#    return tau_data_dict

xm_digits_path = os.path.join(".","images","XiaoMing_Digits")
ob_digits_path = os.path.join(".","images","60 Images")

print("===== LOADING DIGITS =====")

xm_digits, xm_labels = utils.load_image_data(xm_digits_path,side=286, padding=57)
ob_digits, ob_labels = utils.load_image_data(ob_digits_path,side=286, padding=57)

labels = np.concatenate((xm_labels,ob_labels)) 

xm_digits = change_thickness_nosave(xm_digits,min_thickness)
ob_digits = change_thickness_nosave(ob_digits,min_thickness)

combined = np.concatenate((xm_digits,ob_digits),axis=0)