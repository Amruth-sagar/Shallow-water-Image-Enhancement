import numpy as np

def mse(img_1, img_2):

    error = np.mean(np.power(np.subtract(img_1, img_2),2))
    return error