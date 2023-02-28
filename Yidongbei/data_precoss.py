# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年07月25日
"""
# from pip._internal import main
# main(['install', 'PyWavelets'])
import torch
import numpy as np#用于将张量转化为数组，进行除法
import pywt
import matplotlib.pyplot as plt
import os
from scipy.signal import stft,cwt
from PIL import Image

def read_image(data_file,data_type):

    f = Image.open(data_file)
    if data_type!='gray' and data_type!='GAF':
        f = f.resize((224,224))
        img = f.convert('RGB')
        img = np.array(img,dtype=np.float32)/255.0
        return torch.tensor(img.transpose((2,0,1)),dtype=torch.float32)
    elif data_type == 'GAF':
        img = f.convert('RGB')
        img = np.array(img,dtype=np.float32)/255.0
        return torch.tensor(img.transpose((2,0,1)),dtype=torch.float32)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def data2cwt(data_file):
    data = np.loadtxt(data_file)
    sampling_rate = 4096
    wavename = "cgau8"  # 小波函数
    totalscal = 256  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    t = np.arange(len(data))
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)  # 连续小波变换模块
    # ——画图——
    # 尺寸
    figure = plt.figure(figsize=(3.1, 3.08),dpi=200)
    plt.contourf(t, frequencies, abs(cwtmatr), cmap='jet')
    # 关闭坐标轴
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    buf = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = image.resize((224, 224))
    image = np.asarray(image)[:,:,:3]
    plt.close('all')
    return image
def data2stft(data_file):
    data = np.loadtxt(data_file)
    fs = 4096
    window = 'hann'
    n = 256
    f, t, Z = stft(data, fs=fs, window=window, nperseg=n)
    figure = plt.figure(figsize=(3.1, 3.08),dpi=200)
    plt.pcolormesh(t, f, np.abs(Z),shading='auto', cmap='jet')
    # 关闭坐标轴
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    buf = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = image.resize((224, 224))
    image = np.asarray(image)[:,:,:3]
    plt.close('all')
    return image

def load_data(test_path):
    file_list = os.listdir(test_path)
    data_file = []
    for i in file_list:
        data_file.append(os.path.join(test_path,i))
    data_file.sort(key=lambda x: float(x[:-4].split('\\')[-1]))
    return data_file
def data_processor(file):
    cwt_data = torch.tensor(data2cwt(file),dtype=torch.float32)/255.0
    stft_data = torch.tensor(data2stft(file),dtype=torch.float32)/255.0
    in_data = torch.cat([cwt_data.permute(2,0,1),stft_data.permute(2,0,1)],axis=0)
    return in_data.unsqueeze(0)