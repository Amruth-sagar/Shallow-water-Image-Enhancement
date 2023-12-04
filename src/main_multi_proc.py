import os
import sys
sys.path.insert(0,'color_correction/')
import argparse
import cv2
from matplotlib import pyplot as plt
from contrast_correction.contrast_correction import ContrastCorrection
from color_correction import ColorCorrection
from evaluation.evaluate import *
from skimage import color
import numpy as np
from joblib import Parallel, delayed
parser = argparse.ArgumentParser()

#Initialise the method classes
contrast_correction = ContrastCorrection()
color_correction = ColorCorrection()

# Adding Arguments
parser.add_argument('--single', dest='single', action='store_true', help="Want to check results in single image of your choice/")

parser.add_argument("--img_path", help="Want to check results in single image of your choice/", required=False)

parser.add_argument("--dataset", help="mandatory argument to provide the dataset name", required=False)

parser.add_argument("--gt", help="to evaluate the method on the mentioned dataset the output in ..output/", required=False)

parser.add_argument("--save_output", action='store_true', help="to save the output in ..output/")

parser.add_argument("--do_eval", action='store_true', help="to evaluate the method on the mentioned dataset the output in ..output/") 

def process_single(args, image_path, gt_image_path=None, file_save_path=None):

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
   
    contrast_corrected_image = contrast_correction.process(image)
    
    color_corrected_image = color_correction.color_correct(contrast_corrected_image)

    #IF WE ARE EVALUATING FOR A DATASET
    if(args.do_eval):


        gt_image = cv2.cvtColor(cv2.imread(gt_image_path), cv2.COLOR_BGR2RGB)

    
        if args.save_output:
            image_bgr = cv2.cvtColor(color_corrected_image, cv2.COLOR_RGB2BGR)

            h = image_bgr.shape[0]
            w = image_bgr.shape[1]

            new_img = np.ones((2*h+150, 2*w+150,3),dtype=np.uint8)*255

            new_img[50:50+h, 50:50+w, :] = image[:,:,::-1]
            new_img[50:50+h, 100+w:100+2*w, :] = gt_image[:,:,::-1]
            new_img[100+h:100+2*h, 50:50+w, :] = contrast_corrected_image[:,:,::-1]
            new_img[100+h:100+2*h, 100+w:100+2*w,:] = color_corrected_image[:,:,::-1]

            cv2.imwrite(file_save_path, new_img)

        uciqe_val, psnr_val, mse_val, entropy_val = qualitative_analysis(gt_image, color_corrected_image)
        return uciqe_val, psnr_val, mse_val, entropy_val

    #IF WE ARE CHECKING OUTPUT OF A RANDOM IMAGE
    else:
        if args.save_output:

            image_bgr = cv2.cvtColor(color_corrected_image, cv2.COLOR_RGB2BGR)
            plt.figure(figsize=(40,10), dpi=80)
            
            plt.subplot(1,3,1)
            plt.imshow(image)
            plt.title("Original Image",fontsize= 50)
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.imshow(contrast_corrected_image)
            plt.title("Contrast Corrected Image",fontsize= 50)
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.imshow(color_corrected_image)
            plt.title("Color Corrected Image",fontsize= 50)
            plt.axis("off")

            plt.savefig(file_save_path)


def main():

    args = parser.parse_args()
    dataset_name = args.dataset
    if args.gt:
        gt_name = args.gt

    #output_path = '/run/user/1000/gvfs/google-drive:host=gmail.com,user=seshadrimazumder1997/1tB1Hbhl7Y12RD26tb4NcZneTux7sjgCz'

    output_path_single = '../outputs/'+'single_image'
    os.makedirs(output_path_single, exist_ok=True) 


    
    if not args.single:

        if args.do_eval:
            data_path = '../data/'+dataset_name
            output_path = '../outputs/'+dataset_name
            gt_path = '../data/'+gt_name
            os.makedirs(output_path, exist_ok=True) 
            sum_uciqe_val = 0
            sum_psnr_val = 0
            sum_mse_val = 0 
            sum_entropy_val = 0

            print(data_path)
            no_samples = os.listdir(data_path)
            uciqe_val, psnr_val, mse_val, entropy_val = zip(*Parallel(n_jobs=3)(delayed(process_single)(args, os.path.join(data_path, image_name), os.path.join(gt_path, image_name), file_save_path=os.path.join(output_path,image_name)) for image_name in no_samples))
            print(f"UCIQE VAL : {np.mean(list(uciqe_val))}, PSNR_VAL : {np.mean(list(psnr_val))}, MSE_VAL : {np.mean(list(mse_val))}, ENTROPY_VAL : {np.mean(list(entropy_val))}")

        else :

            no_samples = os.listdir(data_path)
            zip(*Parallel(n_jobs=3)(delayed(process_single)(args, os.path.join(data_path, image_name), file_save_path=os.path.join(output_path,image_name)) for image_name in no_samples))

    else:
        path_single_out = os.path.join(output_path_single, "output.png")
        process_single(args, args.img_path, file_save_path=path_single_out)

main()



