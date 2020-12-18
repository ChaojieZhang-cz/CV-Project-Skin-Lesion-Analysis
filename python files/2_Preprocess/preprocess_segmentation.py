import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
import scipy.misc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2
from glob import glob
from tqdm import tqdm

seed = 0
np.random.seed(seed)
torch.manual_seed(seed);


device = torch.device('cpu')


images_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/segmentation/ISIC2018_Task1-2_Training_Input/'
new_images_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/segmentation/images_preprocessed/'

mask_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/segmentation/ISIC2018_Task1_Training_GroundTruth/'
new_mask_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/segmentation/mask_preprocessed/'



files = []
for filename in os.listdir(images_path):
    if '.jpg' in filename:
        files.append(filename)
df = {'image':files}
df = pd.DataFrame(df)
df['mask'] = df['image'].apply(lambda x: x[:-4]+'_segmentation.png')


def change_size(image,mask):
    img = cv2.medianBlur(image,5)
    b=cv2.threshold(img,15,255,cv2.THRESH_BINARY)
    binary_image=b[1]
    binary_image=cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
    
 
    x=binary_image.shape[0]
    y=binary_image.shape[1]
    edges_x=[]
    edges_y=[]
    for i in range(x):
        for j in range(y):
            if binary_image[i][j]==255:
                edges_x.append(i)
                edges_y.append(j)
 
    left=min(edges_x)
    right=max(edges_x)
    width=right-left
    bottom=min(edges_y)
    top=max(edges_y)
    height=top-bottom
 
    pre1_picture = image[left:left+width,bottom:bottom+height]
    new_mask = mask[left:left+width,bottom:bottom+height]
    return pre1_picture,new_mask



def shade_of_gray_cc(img, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)
    
    return img.astype(img_dtype)


target_size = 1024
for file in df.index:
    image_name = df.loc[file,'image']
    mask_name = df.loc[file,'mask']
    
    i_path = images_path+image_name
    m_path = mask_path+mask_name
    
    sample = cv2.imread(i_path)
    sample_mask = cv2.imread(m_path,cv2.IMREAD_GRAYSCALE)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample,sample_mask = change_size(sample,sample_mask)
    
    
    sample = shade_of_gray_cc(sample)
    resize_ratio = target_size/min(sample.shape[0],sample.shape[1])
    sample = cv2.resize(sample,dsize=None,fx=resize_ratio,fy=resize_ratio,interpolation=cv2.INTER_AREA)
    new_sample_path = new_images_path+image_name
    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
    cv2.imwrite(new_sample_path,sample)
    
    
    sample_mask = cv2.resize(sample_mask,dsize=None,fx=resize_ratio,fy=resize_ratio,interpolation=cv2.INTER_NEAREST)
    new_sample_mask_path = new_mask_path + mask_name
    cv2.imwrite(new_sample_mask_path,sample_mask)
    
    print(file)





