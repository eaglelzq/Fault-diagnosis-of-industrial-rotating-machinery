# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年09月01日
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
def read_image(data_file,data_type):

    f = Image.open(data_file)
    if data_type!='gray':
        f = f.resize((224,224))
        img = f.convert('RGB')
        img = np.array(img,dtype=np.float32)/255.0
        return torch.tensor(img.transpose((2,0,1)),dtype=torch.float32)
    else:
        img = np.array(f,dtype=np.float32)/255.0
        return torch.tensor(img,dtype=torch.float32)
class Train_Dataset(Dataset):
    def __init__(self,dir_path):
        super(Train_Dataset, self).__init__()
        csv_data = pd.read_csv(dir_path)
        self.cwt_file = csv_data['cwt_data'].values.tolist()
        self.stft_file = csv_data['stft_data'].values.tolist()
        self.label = torch.tensor(csv_data['label'].values,dtype=torch.int64)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        cwt_data = read_image(self.cwt_file[item],'cwt')
        stft_data = read_image(self.stft_file[item],'stft')

        data = torch.cat([cwt_data,stft_data],axis=0)
        label = self.label[item]


        return data.to('cuda'), label.to('cuda')


class Test_Dataset(Dataset):
    def __init__(self,dir_path):
        super(Test_Dataset, self).__init__()
        csv_data = pd.read_csv(dir_path)
        self.cwt_file = csv_data['cwt_data'].values.tolist()
        self.stft_file = csv_data['stft_data'].values.tolist()
        self.label = torch.tensor(csv_data['label'].values, dtype=torch.int64)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        cwt_data = read_image(self.cwt_file[item],'cwt')
        stft_data = read_image(self.stft_file[item],'stft')

        data = torch.cat([cwt_data,stft_data],axis=0)
        label = self.label[item]


        return data.to('cuda'), label.to('cuda')


class Target_Dataset(Dataset):
    def __init__(self,data_dir):
        super(Target_Dataset, self).__init__()
        self.cwt_file = [os.path.join(os.path.join(data_dir,'cwt'),i) for i in os.listdir(os.path.join(data_dir,'cwt'))]
        self.stft_file = [os.path.join(os.path.join(data_dir,'stft'),i) for i in os.listdir(os.path.join(data_dir,'stft'))]
    def __len__(self):
        return len(self.cwt_file)
    def __getitem__(self, item):
        cwt_data = read_image(self.cwt_file[item],'cwt')
        stft_data = read_image(self.stft_file[item],'stft')
        in_data = torch.cat([cwt_data,stft_data],axis=0)
        return in_data.to('cuda')