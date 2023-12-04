from evaluation.uciqe_metric import *
from evaluation.psnr_metric import *
from evaluation.mse_metric import *
from evaluation.entropy_metric import *

def qualitative_analysis(gt_img, processed_img):

    uciqe_val = 0.0
    psnr_val = 0.0
    mse_val = 0.0
    entropy_val = 0.0

    uciqe_val = uciqe(processed_img)
    psnr_val = psnr(gt_img, processed_img)
    mse_val = mse(gt_img, processed_img)
    entropy_val = calculate_entropy(processed_img)

    return uciqe_val, psnr_val, mse_val, entropy_val

