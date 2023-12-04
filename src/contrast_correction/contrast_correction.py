import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.stats import mode


'''
Flow ->
r,g,b = G_B_color_eq(image)
(I_min_red, I_max_red),(I_min_green,I_max_green),(I_min_blue,I_max_blue) = RGHS_Imin_Imax(r,g,b)
(O_min_red, O_max_red),(O_min_green,O_max_green),(O_min_blue,O_max_blue) = RGHS_Omin_Omax
Output = RGHS(I_mins, I_maxs, O_mins, O_maxs)
'''

class ContrastCorrection:
    # storing the histograms for image channels
    def __init__(self):
        self.hist_red = []
        self.hist_green = []
        self.hist_blue = []
        
        
    def generate_hist(self,img):
        flatten_img = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                flatten_img.append(img[i,j])

        bins = 256
        histogram = np.zeros(bins)
        for pixel in flatten_img:
            histogram[pixel] += 1
        return histogram 


    ''' This method decomposes a color image into R G B channels,
     and apply color equalization on Green and Blue channel'''

    def G_B_color_eq(self,image):
        M = image.shape[0]
        N = image.shape[1]

        # decompose in Red, Green, and Blue channels
        red_im = image[:,:,0]
        green_im = image[:,:,1]  
        blue_im = image[:,:,2] 

        # calculate coef_g
        green_sum = green_im.sum()
        green_avg = green_sum/(255*M*N)
        coef_g = 0.5/green_avg

        # calculate coef_b
        blue_sum = blue_im.sum()
        blue_avg = blue_sum/(255*M*N)
        coef_b = 0.5/blue_avg

        # correct the green and blue channel by multiplying the coefficient with the corresponding channel
        green_im = green_im*coef_g
        blue_im = blue_im*coef_b

        green_im = np.clip(green_im, 0, 255)
        blue_im = np.clip(blue_im, 0, 255)
        
        return np.dstack((red_im, green_im.astype(np.uint8), blue_im.astype(np.uint8)))


    ''' This method calculates the parameters I_min and I_max for 
    adaptive relative global histogram stretching'''

    def RGHS_Imin_Imax(self,red_im, green_im, blue_im):

        # sort the pixels for all the 3 channels
        sorted_pixels_r = np.sort(red_im.flatten())
        sorted_pixels_g = np.sort(green_im.flatten())
        sorted_pixels_b = np.sort(blue_im.flatten())

        # generate histograms for R G B channels 
        self.hist_red = self.generate_hist(red_im)
        self.hist_green = self.generate_hist(green_im)
        self.hist_blue = self.generate_hist(blue_im)

        '''
        separate the pixels whose values are in the smallest 0.1% of the total number on the left side
        of the Mode of the histogram. That gives the I_min for histogram stretching.
        '''
        def get_I_min(sorted_pixels, hist):
            a = np.argmax(hist)
            index_left = np.where(sorted_pixels==a)[0][0]
            I_min = sorted_pixels[int(index_left*0.001)]
            return I_min

        '''
        separate the pixels whose values are in the highest 0.1% of the total number on the right side
        of the Mode of the histogram. That gives the I_max for histogram stretching.
        '''   
        def get_I_max(sorted_pixels,hist):
            length = len(sorted_pixels)
            a = np.argmax(hist)
            index_right = np.where(sorted_pixels==a)[0][0]
            I_max = sorted_pixels[int(-(length - index_right) * 0.001)]
            return I_max

        I_min_red = get_I_min(sorted_pixels_r, self.hist_red)
        I_max_red = get_I_max(sorted_pixels_r, self.hist_red)

        I_min_green = get_I_min(sorted_pixels_g, self.hist_green)
        I_max_green = get_I_max(sorted_pixels_g, self.hist_green)

        I_min_blue = get_I_min(sorted_pixels_b, self.hist_blue)
        I_max_blue = get_I_max(sorted_pixels_b, self.hist_blue)
        
        return [(I_min_red, I_max_red),(I_min_green,I_max_green),(I_min_blue,I_max_blue)]


    def RGHS_Omin_Omax(self, I_min_max):


        histograms = np.array([self.hist_red, self.hist_green, self.hist_blue])

        # Mode and STD of channels
        mode_channels = mode(histograms,axis=1)[0].ravel()
        std_channels = np.sqrt((4-np.pi)/2) * mode_channels

        # K_values and Normalized resudual energy ratios of R,G,B channels
        k_channels = np.array([1.1,0.9,0.9])
        N_rer_channels = np.array([0.83,0.95,0.97])

        # Distance between the scene and the camera (in metres)
        distance = 3
        # Omin = mode - std
        Omin_channels = mode_channels - std_channels
        Omax_channels = (mode_channels+1)/(k_channels*(N_rer_channels**distance))
        

        for i, I_range in enumerate(I_min_max):
            if Omax_channels[i]<I_min_max[i][1] or Omax_channels[i]>255:
                Omax_channels[i] = 255
        
        return [(Omin_channels[0],Omax_channels[0]),
                (Omin_channels[1],Omax_channels[1]),
                (Omin_channels[2],Omax_channels[2])]
    
    def RGHS(self, image):
        
        I_min_max = self.RGHS_Imin_Imax(image[:,:,0], image[:,:,1], image[:,:,2])
        O_min_max = self.RGHS_Omin_Omax(I_min_max)
        image = image.astype(np.float)
        
        # Contrast stretching on all channels
        for k in range(3):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i,j,k] = ((image[i,j,k] - I_min_max[k][0]) * 
                                    ((O_min_max[k][1] - O_min_max[k][0])/(I_min_max[k][1] - I_min_max[k][0])) + 
                                    O_min_max[k][0])
        
        return np.clip(image, 0, 255).astype(np.uint8)
                
    
    def bilateral_filter(self, image_RGB, kernel_size, sigma_spatial, sigma_range):
        return cv2.bilateralFilter(image_RGB, kernel_size, sigmaSpace = sigma_spatial, sigmaColor = sigma_range)
    
    
    def process(self, image, kernel_size = 5, sigma_spatial =0.9 , sigma_range = 0.1):
            
        # Green Blue color equalization
        GB_color_eq_image = self.G_B_color_eq(image)
        # Global histogram stretching
        processed_image = self.RGHS(GB_color_eq_image)
        # Bilateral filter to reduce noise
        processed_image = self.bilateral_filter(processed_image, kernel_size, sigma_spatial, sigma_range)
        
        return processed_image


