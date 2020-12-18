import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
import scipy.misc
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

seed = 0
np.random.seed(seed)
torch.manual_seed(seed);


device = torch.device('cuda')


class DataAugmentation:
    def __init__(self,input_size=572):
        self.input_size = input_size
        pass
    
    def randomFlip(self,image, label):
        if random.randint(0, 1) == 1:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if random.randint(0, 1) == 1:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        return image,label
    
    def randomRotation(self,image, label, mode=Image.BICUBIC):
        if random.randint(0, 1) == 1:
            random_angle = np.random.randint(1, 360)
            image = image.rotate(random_angle, mode)
            label = label.rotate(random_angle, Image.NEAREST)
        return image,label
    
    
    def randomResizedCrop(self,image, label, resized_ratio=(0.2,1.0)):
        crop_size = self.input_size
        image_width = image.size[0]
        image_height = image.size[1]
        if random.randint(0, 1) == 1:
            size_resize = random.randint(int(crop_size/resized_ratio[1]+1),int(crop_size/resized_ratio[0]))
        else:
            size_resize = crop_size+1
        image = transforms.Resize(size_resize, interpolation=Image.BICUBIC)(image)
        lable = transforms.Resize(size_resize, interpolation=Image.NEAREST)(label)
        
        
        image_width = image.size[0]
        image_height = image.size[1]
        crop_left = random.randint(0,int(image_width-crop_size-1))
        crop_up = random.randint(0,int(image_height-crop_size-1))
        
        crop_box = (crop_left,crop_up,crop_left+crop_size,crop_up+crop_size)
        image = image.crop(crop_box)
        lable = lable.crop(crop_box)
        
        return image,lable
    
    def test_Resized(self,image, label):
        crop_size = self.input_size
        width = image.size[0]
        height = image.size[1]
        desired_size = max(width,height)
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(image)
        new_mask = Image.new("1", (desired_size, desired_size))
        new_mask.paste(label)
        image = transforms.Resize(crop_size, interpolation=Image.BICUBIC)(new_im)
        lable = transforms.Resize(crop_size, interpolation=Image.NEAREST)(new_mask)
        return image,lable
        
    
    def randomGaussian(self,image, label, mean=0.2, sigma=0.3):
        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.asarray(image).copy()
        img.flags.writeable = True
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label
        
        
    def randomColor(self,image, label):
        image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0)(image)
        return image,label
    
    def Normalization(self,image, label):
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.4373, 0.4434, 0.4725],[0.1201, 0.1231, 0.1052])(image)
        label = transforms.ToTensor()(label)
        return image,label
    
    def randomCover(self,image, label,cover=128):
        if random.randint(0, 1) == 1:
            image_width = image.size(1)
            image_height = image.size(2)

            cover_left = random.randint(0,int(image_width-cover-1))
            cover_up = random.randint(0,int(image_height-cover-1))

            image[:,cover_left:(cover_left+cover),cover_up:(cover_up+cover)]=0
            label[:,cover_left:(cover_left+cover),cover_up:(cover_up+cover)]=0
        return image,label
    
    def TrainAugmentation(self,image,label):
        new_image,new_label = self.randomFlip(image,label)
        new_image,new_label = self.randomRotation(new_image,new_label)
        new_image,new_label = self.randomResizedCrop(new_image,new_label)
        new_image,new_label = self.randomGaussian(new_image,new_label)
        new_image,new_label = self.randomColor(new_image,new_label)
        new_image,new_label = self.Normalization(new_image,new_label)
        #new_image,new_label = self.randomCover(new_image,new_label)
        return new_image,new_label
    
    def TestProcess(self,image,label):
        new_image,new_label = self.test_Resized(image, label)
        new_image,new_label = self.Normalization(new_image,new_label)
        return new_image,new_label
        
        
        
class image_dataset(Dataset):
    def __init__(self, df_path, train = False):
        self.df = pd.read_csv(df_path)
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        image_name = self.df.iloc[idx]['image']
        image_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/segmentation/images_preprocessed/'\
        + image_name
        mask_name = self.df.iloc[idx]['mask']
        mask_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/segmentation/mask_preprocessed/'\
        + mask_name
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        if self.train:
            new_image,new_label = DataAugmentation().TrainAugmentation(image,mask)
        else:
            new_image,new_label = DataAugmentation().TestProcess(image,mask)
        
        sample={'x':new_image,'y':new_label}
        return sample
        
        
train_df_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Jupyter Notebook/seg_train.csv'
val_df_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Jupyter Notebook/seg_val.csv'
test_df_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Jupyter Notebook/seg_test.csv'
BATCH_SIZE = 4
train_loader = DataLoader(image_dataset(train_df_path,train = True), batch_size=BATCH_SIZE, shuffle=True,num_workers=64)
val_loader = DataLoader(image_dataset(val_df_path), batch_size=BATCH_SIZE, shuffle=True,num_workers=64)
test_loader = DataLoader(image_dataset(test_df_path), batch_size=BATCH_SIZE, shuffle=False,num_workers=64)




resnet18 = models.resnet18()
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, 9)
        
    def forward(self, image, data):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

resnet18 = MyModel()
path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Jupyter Notebook/py files/ResNet18/checkpoint_Model_ResNet18.pt'
resnet18.load_state_dict(torch.load(path,map_location='cpu')['best_model_wts'])



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        
        
class U_ResNet18(nn.Module):
    def __init__(self,bilinear=True):
        super(U_ResNet18, self).__init__()
        self.bilinear = bilinear

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        
        self.up1_conv1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.up1 = Up(1024, 256, bilinear)
        self.up2_conv1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.up2 = Up(512, 128, bilinear)
        self.up3_conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.up3 = Up(256, 64, bilinear)
        self.up4_conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.up4 = Up(128, 32, bilinear)
        self.up5_conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.up5_conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.up5 = Up(64, 32, bilinear)
        
        self.outc = OutConv(32, 1)
        

    def forward(self, x):
        x0 = x
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x
        #print(x.size())
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x
        #print(x.size())
        x = self.layer2(x)
        x3 = x
        #print(x.size())
        x = self.layer3(x)
        x4 = x
        #print(x.size())
        x = self.layer4(x)
        x5 = x
        #print(x.size())
        
        x4 = self.up1_conv1(x4)
        x = self.up1(x5, x4)
        #print(x.size())
        x3 = self.up2_conv1(x3)
        x = self.up2(x, x3)
        #print(x.size())
        x2 = self.up3_conv1(x2)
        x = self.up3(x, x2)
        #print(x.size())
        x1 = self.up4_conv1(x1)
        x = self.up4(x, x1)
        #print(x.size())
        x0 = self.up5_conv1(x0)
        x0 = self.up5_conv2(x0)
        x = self.up5(x, x0)
        #print(x.size())
        x = self.outc(x)
        return x
        
model = U_ResNet18()




class BCE_IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCE_IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        IoU_loss  = 1 - IoU
        
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        BCE_IoU_loss = IoU_loss+BCE_loss
                
        return BCE_IoU_loss  
        

class IoU_metric(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_metric, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)  
        inputs = torch.round(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU
        
        
lr = 0.001
loss_fn = BCE_IoULoss()
metric = IoU_metric()        


def train(model, train_loader=train_loader, val_loader=val_loader, learning_rate=5e-5, num_epoch=1000,\
          lr=lr,loss_fn=loss_fn,metric=metric):
    start_time = time.time()
    
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=2)
    
    
    train_loss_return = []
    train_iou_return = []
    val_loss_return = []
    val_iou_return = []
    best_iou = 0
    
    for epoch in range(num_epoch):
        # Training steps
        train_loss_list = []
        train_iou_list = []
        
        model.train()
        for i, (sample) in enumerate(train_loader):
            image = sample['x'].to(device)
            labels = sample['y'].to(device)
            outputs = model(image)
            model.zero_grad()
            loss = loss_fn(outputs, labels)
            iou = metric(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            train_iou_list.append(iou.item())
            
        # report performance
        train_iou_every_epoch = np.average(train_iou_list)
        train_iou_return.append(train_iou_every_epoch)
        train_loss_every_epoch = np.average(train_loss_list)
        train_loss_return.append(train_loss_every_epoch)
        print('----------Epoch{:2d}/{:2d}----------'.format(epoch+1,num_epoch))
        print('Train set | Loss: {:6.6f} | IoU: {:4.4f}% '.format(train_loss_every_epoch, train_iou_every_epoch*100))
        
        # Evaluate after every epochh
        model.eval()
        val_loss_list = []
        val_iou_list = []
        
        
        with torch.no_grad():
            for i, (sample) in enumerate(val_loader):
                image = sample['x'].to(device)
                labels = sample['y'].to(device)
                outputs = model(image)
                loss = loss_fn(outputs, labels)
                iou = metric(outputs, labels)
                val_loss_list.append(loss.item())
                val_iou_list.append(iou.item())
            # report performance
            val_iou_every_epoch = np.average(val_iou_list)
            val_iou_return.append(val_iou_every_epoch)
            val_loss_every_epoch = np.average(val_loss_list)
            val_loss_return.append(val_loss_every_epoch)
            
            if val_iou_every_epoch > best_iou:
                best_iou = val_iou_every_epoch
                best_model_wts = model.state_dict()
            
            save_model(model,train_loss_return,train_iou_return,val_loss_return,val_iou_return,best_model_wts)
            
            elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
            print('Test set | Loss: {:6.6f} | IoU: {:4.4f}% | time elapse: {:>9}'\
                  .format(val_loss_every_epoch, val_iou_every_epoch*100,elapse))
    return model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts

def save_model(model,train_loss_return,train_iou_return,val_loss_return,val_iou_return,best_model_wts):
    state = {'best_model_wts':best_model_wts, 'model':model, \
             'train_loss':train_loss_return, 'train_iou':train_iou_return,\
             'val_loss':val_loss_return, 'val_iou':val_iou_return}
    torch.save(state, 'checkpoint_UNet.pt')
    return None
    
    
model = model.to(device)
train(model)











