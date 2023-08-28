# download and extract
# !wget https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_0-1999_72_imgs.zip
# !wget https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_2000-3999_72_imgs.zip
# !wget https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_4000-5999_72_imgs.zip
# !wget https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_6000-7999_72_imgs.zip
# !wget https://facesyntheticspubwedata.blob.core.windows.net/wacv-2023/subjects_8000-9999_72_imgs.zip

Config = {
    'batch_size': {
        'values' : [64]
        },
    'epochs':{
        'values': [25]
    },
    'optimizer': {
        'values' : ['adam','Madgrad']
    },
    'lr': {
        'values': [3e-5]
    },
    'weight_decay':{
        'values': [1e-3]
    },
    'embedding_dimension':{
        'values': [8000]
    }
}
sweep_config = {
    'method': 'grid'
    }

metric = {
    'name': 'Best_AUC',
    'goal': 'maximize'   
    }
sweep_config['metric'] = metric
sweep_config['parameters'] = Config

import os
from tqdm import tqdm
import torch
from madgrad import MADGRAD
import cv2
import torch.nn as nn
from torchvision import transforms
from sup_con import SupConLoss
from model import embed
import numpy as np
import random

path_to_data = ''
paths = []
for i in tqdm(os.listdir('path_to_data')):
    for j in os.listdir('path_to_data'+str(i)):
        paths.append('path_to_data'+str(i)+'/'+str(j))
random.shuffle(paths)

class dataset(torch.utils.data.Dataset):
    def __init__(self, paths, transforms = None):
        super().__init__()
        self.paths = paths
        self.transforms = transforms
    def __len__(self):
        return len(self.paths)
    def __getitem__(self,idx):
        img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)
        label = int(self.paths[idx].split('/')[4])
        if self.transforms == None:
            return img,label
        return self.transforms(img),label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((125,125)),
    transforms.RandomCrop(100),
    transforms.ToTensor()
])
transform_1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((125,125)),
    transforms.RandomCrop(75),
    transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
    transforms.Resize((100,100)),
    transforms.ToTensor()
])

def train(model, device, optim, loss, dataloader_train_1, dataloader_train_2):
    model.train()
    epoch_loss = []
    for (img1,labels),(img2,_) in tqdm(zip(dataloader_train_1, dataloader_train_2), total = len(dataloader_train_1)):
        img1, img2 = img1.to(device),img2.to(device)
        imgs = torch.cat([img1,img2], dim = 0)
        embeds = model(imgs)
        e1, e2 = torch.split(embeds, [labels.shape[0], labels.shape[0]], dim=0)
        embeds = torch.cat([e1.unsqueeze(1), e2.unsqueeze(1)], dim=1)
        batch_loss = loss(embeds,labels)
        epoch_loss.append(batch_loss.detach().cpu().numpy())
        batch_loss.backward()
        optim.step()
    return np.mean(epoch_loss)

def val(model, device, loss, dataloader_val_1, dataloader_val_2):
    model.eval()
    epoch_loss = []
    for (img1,labels),(img2,_) in tqdm(zip(dataloader_val_1, dataloader_val_2), total = len(dataloader_val_1)):
        img1, img2 = img1.to(device),img2.to(device)
        imgs = torch.cat([img1,img2], dim = 0)
        embeds = model(imgs)
        e1, e2 = torch.split(embeds, [labels.shape[0], labels.shape[0]], dim=0)
        embeds = torch.cat([e1.unsqueeze(1), e2.unsqueeze(1)], dim=1)
        batch_loss = loss(embeds,labels)
        epoch_loss.append(batch_loss.detach().cpu().numpy())
    return np.mean(epoch_loss)

sweep_id = wandb.sweep(sweep_config, project="", entity='')
def run(config = None):
    with wandb.init(config = Config):
        config = wandb.config 

    batch_size = config.batch_size
    dataloader_train_1 = torch.utils.data.DataLoader(dataset(paths[:int(len(paths)*0.8)],transform),batch_size = batch_size, shuffle = False)
    dataloader_train_2 = torch.utils.data.DataLoader(dataset(paths[:int(len(paths)*0.8)],transform_1),batch_size = batch_size, shuffle = False)
    dataloader_val_1 = torch.utils.data.DataLoader(dataset(paths[int(len(paths)*0.8):],transform),batch_size = batch_size, shuffle = False)
    dataloader_val_2 = torch.utils.data.DataLoader(dataset(paths[int(len(paths)*0.8):],transform_1),batch_size = batch_size, shuffle = False)
    
    model_embed =  embed()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss = SupConLoss()
    optim = torch.optim.Adam(model_embed.parameters(), lr = config.lr)
    optim = torch.optim.Adam(model_embed.parameters(), lr = config.lr, weight_decay=config.weight_decay) if config.optimizer == 'Adam' else MADGRAD(model_embed.parameters(), lr = config.lr, weight_decay=config.weight_decay)

    epochs = config.epochs
    best_val_loss = float('inf')
    best_model = embed(config)

    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}/{epochs}')
        train_loss = train(model_embed,device,optim,loss)
        val_loss = val(model_embed,device,loss)
        if epoch%10 == 0:
            torch.save(model_embed,f'model_'+str(epoch)+'.pt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model_embed
        print(f'train loss: {train_loss}\t\t val loss: {val_loss}')
        print('-------------------------------------------------------------')

    torch.save(model_embed,'model_embed_last.pt')
    torch.save(best_model,'model_embed_best.pt')
wandb.agent(sweep_id, run)