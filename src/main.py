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
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Adding Arguments
parser.add_argument("--dataset", help="mandatory argument to provide the dataset name")

parser.add_argument("--save_output", help="to save the output in ..output/", type=bool, default=False)


args = parser.parse_args()
dataset_name = args.dataset

data_path = '../data/'+dataset_name
output_path = '../outputs/'+dataset_name

os.makedirs(output_path, exist_ok=True)

#Initialise the method classes
contrast_correction = ContrastCorrection()
color_correction = ColorCorrection()

no_samples = os.listdir(data_path)
no_samples.sort()

for i,image_name in enumerate(no_samples):

    image_path = os.path.join(data_path, image_name)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    
    contrast_corrected_image = contrast_correction.process(image)
    
    color_corrected_image = color_correction.color_correct(contrast_corrected_image)

    print('Image Number: {}'.format(i))
    
    if args.save_output:
        image_bgr = cv2.cvtColor(color_corrected_image, cv2.COLOR_RGB2BGR)
        h = image_bgr.shape[0]
        w = image_bgr.shape[1]
        out_image_path = os.path.join(output_path,image_name)[:-3]+"jpeg"
        cv2.imwrite(out_image_path, image_bgr)
       

