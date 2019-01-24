import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

def load_images(img_dir_path):
    images = os.listdir(img_dir_path)
    img_vec_list = []
    for img in images:
        img_arr = cv2.imread(os.path.join(img_dir_path,img), 0)
        img_inv = cv2.bitwise_not(img_arr)
        img_resized = cv2.resize(img_inv, (28,28))
        img_vec_list.append(img_resized)
    
    return img_vec_list


imagepath = os.path.join(".","images","60 Images")
imgs = load_images(imagepath)

(X_train,y_train),(x_test,y_test) = mnist.load_data()

ones = np.where(y_train == 1)
print(ones)

plt.figure()
plt.imshow(imgs[10])
plt.figure()
plt.imshow(X_train[3])
plt.show()

