import numpy as np
from skimage.color import lab2rgb

#Function to convert CIE Lab to RGB image
def CIE_Lab_to_RGB(Lab):
    return (lab2rgb(Lab)*255).astype('uint8')
