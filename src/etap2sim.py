import utils
import numpy as np
from matplotlib import pyplot as plt
import os

path = os.path.join(".","images","60 Images")
digits,labels = utils.load_image_data(path, radius = 4)

num = digits[10,:]
plt.imshow(num.reshape(28,28))
plt.show()