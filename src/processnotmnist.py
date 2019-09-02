import numpy as np
import utils
import utils.digitutils as dutils
import os
import cv2

def create_not_mnist():

    def load_letter(letter_path):
        def has_shape(img_arr):
            h,w = img_arr.shape
            return h != 0 and w != 0

        def load_image(img_path,bw):
            img = cv2.imread(img_path, 0)
            if bw:
                _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return img

        images = os.listdir(letter_path)
        img_list = []
        for img in images:
            img_arr = load_image(os.path.join(letter_path,img),True)
            if (not (img_arr is None)):
                img_arr = dutils.unpad_img(img_arr)
                if(has_shape(img_arr)):
                    img_list.append(img_arr)

        img_list = list(map(dutils.unpad_img, img_list))

        img_list = list(map(
            lambda img: dutils.center_box_image(dutils.resize_image(img, 20), 20, 4)
        ,img_list))

        return np.array(img_list)


    path = os.path.join('notMNIST_small')
    letters_dict = {}
    letters = ['A','B','C','D','E','F','G','H','I','J']
    for letter in letters:
        l_list = load_letter(os.path.join(path,letter))
        count,_,_ = l_list.shape
        l_index = np.random.permutation(count)
        letters_dict[letter] = l_list[l_index[:100]]       

    return letters_dict
not_mnist = create_not_mnist()
utils.save_processed_data(not_mnist,'notMNIST1000')
