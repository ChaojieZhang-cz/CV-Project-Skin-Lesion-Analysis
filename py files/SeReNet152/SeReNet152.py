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
from efficientnet_pytorch import EfficientNet
import pretrainedmodels

seed = 0
np.random.seed(seed)
torch.manual_seed(seed);

device = torch.device('cuda')


train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.RandomVerticalFlip())
transforms.RandomRotation(180, expand=True)
train_transform.transforms.append(transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(1.0,1.0)))
train_transform.transforms.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0))
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(transforms.Normalize([0.4373, 0.4434, 0.4725],[0.1201, 0.1231, 0.1052]))
train_transform.transforms.append(transforms.RandomErasing())


test_transform = transforms.Compose([])
test_transform.transforms.append(transforms.RandomResizedCrop(224, scale=(1.0, 1.0), ratio=(1,1)))
test_transform.transforms.append(transforms.ToTensor())
test_transform.transforms.append(transforms.Normalize([0.4373, 0.4434, 0.4725],[0.1201, 0.1231, 0.1052]))



class image_dataset(Dataset):
    def __init__(self, df_path, train = False):
        self.df = pd.read_csv(df_path)
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        image_name = self.df.iloc[idx]['image']
        image_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Data/classification/images_preprocessed/'\
        + image_name + '.jpg'
        image = Image.open(image_path)
        
        if self.train:
            image_tensor = train_transform(image)
        else:
            image_tensor = test_transform(image)
            
        metadata = self.df.loc[idx][['sex_index','age_index','anterior torso', 'head/neck', 'lateral torso',\
                                     'lower extremity','oral/genital', 'palms/soles', 'posterior torso',\
                                     'upper extremity']].values.astype('float')
        metadata = torch.tensor(metadata, dtype=torch.float)
        label = self.df.loc[idx][['MEL','NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC','UNK']].values.astype('float')
        label = torch.tensor(label, dtype=torch.float)
        label = label.data.max(-1)[1]
        sample = {'x1': image_tensor, 'x2':metadata,'y': label}
        
        return sample
        
        
train_df_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Jupyter Notebook/train.csv'
val_df_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Jupyter Notebook/val.csv'
test_df_path = '/scratch/cz2064/myjupyter/Computer Vision/Project/Jupyter Notebook/test.csv'
BATCH_SIZE = 16
train_loader = DataLoader(image_dataset(train_df_path,train = True), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(image_dataset(val_df_path), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(image_dataset(test_df_path), batch_size=BATCH_SIZE, shuffle=True)


model_name = 'se_resnet152'
se_resnet152 = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = se_resnet152
      
        self.fc = nn.Linear(1000, 9)
        
    def forward(self, image, data):
        x1 = self.cnn(image)
        #x2 = F.relu(self.fc2(data))
        #x = torch.cat((x1, x2), dim=1)       
        x = self.fc(x1)
        
        return x
        

def train(model, train_loader=train_loader, val_loader=val_loader, learning_rate=5e-5, num_epoch=100):
    start_time = time.time()
    
    distribution = torch.FloatTensor([0.17652477, 0.50707283, 0.13323245, 0.03348905, \
                                  0.10494111, 0.00894796, 0.00986907, 0.02592276, 0.]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=distribution)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    
    train_loss_return = []
    train_acc_return = []
    val_loss_return = []
    val_acc_return = []
    best_acc = 0
    
    for epoch in range(num_epoch):
        # Training steps
        correct = 0
        total = 0
        predictions = []
        truths = []
        model.train()
        train_loss_list = []
        for i, (sample) in enumerate(train_loader):
            image = sample['x1'].to(device)
            data = sample['x2'].to(device)
            labels = sample['y'].to(device)
            outputs = model(image,data)
            pred = outputs.data.max(-1)[1]
            predictions += list(pred.cpu().numpy())
            truths += list(labels.cpu().numpy())
            total += labels.size(0)
            correct += (pred == labels).sum()
            model.zero_grad()
            loss = loss_fn(outputs, labels)
            train_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        # report performance
        acc = (100 * correct / total)
        train_acc_return.append(acc)
        train_loss_every_epoch = np.average(train_loss_list)
        train_loss_return.append(train_loss_every_epoch)
        print('----------Epoch{:2d}/{:2d}----------'.format(epoch+1,num_epoch))
        print('Train set | Loss: {:6.4f} | Accuracy: {:4.2f}% '.format(train_loss_every_epoch, acc))
        
        # Evaluate after every epochh
        correct = 0
        total = 0
        model.eval()
        predictions = []
        truths = []
        val_loss_list = []
        with torch.no_grad():
            for i, (sample) in enumerate(val_loader):
                image = sample['x1'].to(device)
                data = sample['x2'].to(device)
                labels = sample['y'].to(device)
                outputs = model(image,data)
                loss = loss_fn(outputs, labels)
                val_loss_list.append(loss.item())
                pred = outputs.data.max(-1)[1]
                predictions += list(pred.cpu().numpy())
                truths += list(labels.cpu().numpy())
                total += labels.size(0)
                correct += (pred == labels).sum()
            # report performance
            acc = (100 * correct / total)
            val_acc_return.append(acc)
            val_loss_every_epoch = np.average(val_loss_list)
            val_loss_return.append(val_loss_every_epoch)
            if acc > best_acc:
                best_acc = acc
                best_model_wts = model.state_dict()
            save_model(model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts)
            elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
            print('Test set | Loss: {:6.4f} | Accuracy: {:4.2f}% | time elapse: {:>9}'\
                  .format(val_loss_every_epoch, acc,elapse))
    return model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts

def save_model(model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts):
    state = {'best_model_wts':best_model_wts, 'model':model, \
             'train_loss':train_loss_return, 'train_acc':train_acc_return,\
             'val_loss':val_loss_return, 'val_acc':val_acc_return}
    torch.save(state, 'checkpoint_Model_resenet152.pt')
    return None
    

model = MyModel().to(device)
train(model)
