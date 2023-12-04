import numpy as np
from skimage.color import rgb2lab
import cv2

# Function to convert RGB image to CIE Lab
def RGB_to_CIE_Lab(img_RGB):
    Lab = rgb2lab(img_RGB)
    L, a, b = cv2.split(Lab)
    return L,a,b