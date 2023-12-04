
# This is a script for Adaptive stretching of CIE Lab values
import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from rgb2cie import *
from cie2rgb import *
from sklearn.preprocessing import scale
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from skimage import color


# S_model curve function to stretch 'a' and 'b' values
def S_model_curve(x):
    return x * (1.3**(1-np.abs(x/128)))

# Rayleigh Distriution function to stretch 
def rayleigh_distribution(x,a):
    return (x/a**2)* np.exp(-x**2 / (2*a**2))
    
# Adaptive Stretching of 'L' values based on Rayleigh Distribution to range [0,100]
def adaptive_stretching_L(L, height, width):
    O_min = 0
    O_max = 100
    stretched_L = np.zeros(L.shape, 'float64')

    L_flatten = L.flatten()
    L_flatten.sort()
    mode = stats.mode(L_flatten).mode[0]
    mode_index = list(L_flatten).index(mode)
    L_min = L_flatten[int(mode_index * 0.001)]
    L_max = L_flatten[int(-(len(list(L_flatten)) - mode_index) * 0.001)]

    for i in range(height):
        for j in range(width):
            if L[i][j] < L_min:
                stretched_L[i][j] = 0
            elif (L[i][j] > L_max):
                stretched_L[i][j] = 100
            else:
                x = (O_min + (L[i][j] - L_min) * ((O_max - O_min)/(L_max - L_min)))
                stretched_L[i][j] = L[i,j]

    return stretched_L


# Function to stretch 'a' or 'b' values to range [-128,127] according to S-model Curve Function
def adaptive_stretching_ab(a_or_b_values,height,width):
    stretched_a_or_b = np.zeros((height,width), 'float64')
    for i in range(height):
        for j in range(width):
            stretched_a_or_b[i][j] = S_model_curve(a_or_b_values[i][j])
    return stretched_a_or_b

def  LAB_Stretching(img_rgb):

    #Processing of Image
    height = len(img_rgb)
    width = len(img_rgb[0])
    
    all_lab = color.rgb2lab(img_rgb)
    L_min, a_min, b_min = np.min(all_lab, axis=(0, 1))
    L_max, a_max, b_max = np.max(all_lab, axis=(0, 1))

    #Converting RGB to CIE Lab
    L,a,b = RGB_to_CIE_Lab(img_rgb)


    #Stretching
    L_stretched = adaptive_stretching_L(L, height, width)

    a_stretched = adaptive_stretching_ab(a, height, width)

    b_stretched = adaptive_stretching_ab(b, height, width)

    #Merging Back
    lab_img = np.zeros((height, width, 3), 'float64')
    
    lab_img[:, :, 0] = L_stretched
    lab_img[:, :, 1] = a_stretched
    lab_img[:, :, 2] = b_stretched        
    
    img_rgb_color_correction = CIE_Lab_to_RGB(lab_img)

    return img_rgb_color_correction


