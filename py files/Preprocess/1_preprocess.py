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

images_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/classification/ISIC_2019_Training_Input/'
new_images_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/classification/images_preprocessed/'

metadata_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/classification/ISIC_2019_Training_Metadata.csv'

metadata_df = pd.read_csv(metadata_path)

def change_size(image):
    #image=cv2.imread(read_file,1) #读取图片 image_name应该是变量
    img = cv2.medianBlur(image,5) #中值滤波，去除黑色边际中可能含有的噪声干扰
    b=cv2.threshold(img,15,255,cv2.THRESH_BINARY)          #调整裁剪效果
    binary_image=b[1]               #二值图--具有三通道
    binary_image=cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
    #print(binary_image.shape)       #改为单通道
 
    x=binary_image.shape[0]
    #print("高度x=",x)
    y=binary_image.shape[1]
    #print("宽度y=",y)
    edges_x=[]
    edges_y=[]
    for i in range(x):
        for j in range(y):
            if binary_image[i][j]==255:
                edges_x.append(i)
                edges_y.append(j)
 
    left=min(edges_x)               #左边界
    right=max(edges_x)              #右边界
    width=right-left                #宽度
    bottom=min(edges_y)             #底部
    top=max(edges_y)                #顶部
    height=top-bottom               #高度
 
    pre1_picture=image[left:left+width,bottom:bottom+height]        #图片截取
    #print(pre1_picture.shape[:2])
    return pre1_picture                                             #返回图片数据

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
    
    
target_size = 512


for file in metadata_df['image']:
    sample_path = images_path+file+'.jpg'
    sample = cv2.imread(sample_path)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample = change_size(sample)
    sample = shade_of_gray_cc(sample)
    resize_ratio = target_size/min(sample.shape[0],sample.shape[1])
    sample = cv2.resize(sample,dsize=None,fx=resize_ratio,fy=resize_ratio,interpolation=cv2.INTER_AREA)
    new_sample_path = new_images_path+file+'.jpg'
    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
    cv2.imwrite(new_sample_path,sample)
    print(file,' ',sample.shape)