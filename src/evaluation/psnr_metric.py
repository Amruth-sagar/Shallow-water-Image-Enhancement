from evaluation.mse_metric import mse
import math

def psnr(img_1, img_2):

    mse_value = mse(img_1, img_2)
    psnr_value = -10*math.log10(1/mse_value)
    return psnr_value