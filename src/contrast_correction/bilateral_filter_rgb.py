import cv2

def bilateral_filter(image_RGB,kernel_size,sigma_spatial,sigma_range):
    return cv2.bilateralFilter(image_RGB,kernel_size,sigmaSpace = sigma_spatial,sigmaColor = sigma_range)
