import utils
from utils import digitutils as dutils
import os
import numpy as np
import cv2
from console_progressbar import ProgressBar
from multiprocessing import Pool, Lock, Value
import time
import copy
import datetime

#mnist_linethickness = 67.14082725553595

min_thickness = 31
max_thickness = 50

class ProgressReporter(object):
    def __init__(self):
        self.lock = Lock()

    def reset(self,pb):
        self.iterations = Value('i', 0)
        self.pb = pb

    def report_complete(self):
        with self.lock:
            self.iterations.value += 1
            self.pb.print_progress_bar(self.iterations.value)

class CommonWidthAdjuster(object):

    def __init__(self,p_reporter):
        self.reporter = p_reporter

    def reset(self,factor,pb):
        self.reporter.reset(pb)
        self.factor = factor

    def adjust_linewidth(self,img):
        new_img = utils.change_linewidth(img,self.factor)
        self.reporter.report_complete()
        return new_img

class ParallellWidthAdjuster(object):

    def __init__(self, p_reporter):
        self.reporter = p_reporter

    def reset(self,rtau,pb):
        self.reporter.reset(pb)
        self.r_tau = rtau

    def adjust_linewidth(self, img):
        tau = utils.intern_calc_linewidth(img)
        diff = 0.9
        while abs(self.r_tau - tau) > diff:
            r = int(round(self.r_tau - tau))
            r = r if r != 0 else int(np.sign(dt)) # Ensures that r is at least 1 or -1
            #print("{rtau : %s, tau: %s, r = %s}" % (self.r_tau, tau, r))
            img = utils.change_linewidth(img,r)
            tau = utils.intern_calc_linewidth(img)
            #print("new line thickness tau = %s" % tau)
        self.reporter.report_complete()
        return img


p_reporter = ProgressReporter()
cwa = CommonWidthAdjuster(p_reporter)
pwa = ParallellWidthAdjuster(p_reporter)

def scale_down(data):
    downscaled = list(map(
        lambda img: cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC), data
    ))

    return np.array(downscaled)

def wrapper_cwa_linewidth(img):
    return cwa.adjust_linewidth(img)

def wrapper_pwa_linewidth(img):
    return pwa.adjust_linewidth(img)

def change_thickness_helper(data, factor):
    global cwa
    datapoints = data.shape[0]
    pb = ProgressBar(total=datapoints,prefix="processing digits:", length=20, fill="=",zfill="_")
    cwa.reset(factor,pb)
    with Pool(6) as p:
        new_data = p.map(wrapper_cwa_linewidth, data)
        new_data = np.array(new_data)
    return new_data

def change_thickness_individual(data, ref_thickness):
    global pwa

    print("Adjusting the thicknesses to %s" % ref_thickness)
    curr_data = data
    datapoints = data.shape[0]
    pb = ProgressBar(total=datapoints,prefix="processing digits:", length=20, fill="=",zfill="_")
    pwa.reset(ref_thickness, pb)

    with Pool(6) as p:
        curr_data = p.map(wrapper_pwa_linewidth, curr_data)
        curr_data = np.array(curr_data)
    #    print("current Linethickness %s" % utils.calc_linewidth(curr_data))

    return curr_data


def change_thickness(data, ref_thickness):
    print("Adjusting the thicknesses to %s" % ref_thickness)
    diff = 0.5
    curr_data = data
    while abs(ref_thickness - tau) > diff:

        tau = utils.calc_linewidth(curr_data)
        print("current linethickness : %s" % tau)
        dt = ref_thickness - tau
        r = int(round(dt))
        r = r if r != 0 else int(np.sign(dt)) # Ensures that r is at least 1 or -1
        if abs(ref_thickness - tau) > diff:
            print("Adjusting Linewidth with r = %s" % r)
            curr_data = change_thickness_helper(curr_data,r)
    return curr_data


def build_thickness_data(data,min_thickness,max_thickness):
    tau_data_dict = {}

    print("Preparing testing data")
    for t in range(min_thickness,max_thickness + 1):

        changed_data = change_thickness_individual(data, t)
        tau_data_dict[t] = scale_down(changed_data)

    return tau_data_dict

def normalize_digit_thickness(digits):
    global pwa
    tau = utils.calc_linewidth(digits)

    datapoints = digits.shape[0]
    pb = ProgressBar(total=datapoints,prefix="processing digits:", length=20, fill="=",zfill="_")

    pwa.reset(tau,pb)
    print("Normalizing %s digits to tau = %s" % (datapoints,tau))
    with Pool(6) as p:
        d_normalized = p.map(wrapper_pwa_linewidth, digits)
    return np.array(d_normalized)


def load_process_digits(pathlist):

    labels = np.array([])
    combined_data_dict = {}
    for t in range(min_thickness,max_thickness+1):
        combined_data_dict[t] = []

    dataset_index = 0
    t = time.time()
    for p in pathlist:
        dataset_index += 1
        digits_data, labels_data = utils.load_image_data(p,side=200, padding=40)
        labels = np.concatenate((labels,labels_data))
        #digits_data = normalize_digit_thickness(digits_data)
        #digits_data = change_thickness(digits_data,min_thickness)
        tau_data_dict = build_thickness_data(digits_data,min_thickness,max_thickness)
        processed_digit_copy = copy.deepcopy(tau_data_dict)
        processed_digit_copy["labels"] = labels_data
        filename = 'Dataset-%i-save3050' % dataset_index
        utils.save_processed_data(processed_digit_copy,filename)

        for k in tau_data_dict.keys():
            d = tau_data_dict[k]
            combined_data_dict[k].append(d)
        print(labels.shape)


    for t in combined_data_dict.keys():
        combined_data_dict[t] = np.concatenate(combined_data_dict[t],axis = 0)
    elapsed = time.time() - t
    print("This ran for : %s" % (datetime.timedelta(seconds = elapsed)))
    return combined_data_dict, labels

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

data_dict, labels = load_process_digits(digit_paths)
print(data_dict)
data_dict['labels'] = labels
utils.save_processed_data(data_dict,'combined_testing_data3150')

print("Script done running")
