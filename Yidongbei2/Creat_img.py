# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年09月01日
"""
import pandas as pd
import torch

import pywt
import os
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt

def data2img(datafile,imgtype,save_root):
    if imgtype=='cwt':
        img = data2cwt(datafile,save_root)
    elif imgtype=='stft':
        img = data2stft(datafile,save_root)
    return img

def data2cwt(data_file,save_root):
    """
    将4096长度的振动信号通过小波变换转换为时频图
    """
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
    plt.figure(figsize=(4, 4))
    plt.contourf(t, frequencies, abs(cwtmatr), cmap='jet')
    # 关闭坐标轴
    plt.axis('off')
    # plt.show()
    # break
    mid_dir = data_file.split('\\')[-2]+'_'+data_file.split('\\')[-1].split('.')[0]
    # 保存图像
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    savefile = save_root + '\\' + mid_dir
    plt.savefig(savefile, bbox_inches='tight', pad_inches=0,dpi=200)
    plt.close('all')


def data2stft(data_file,save_root):
    """
    将长度为4096的振动信号通过短时傅里叶变换转换为时频图
    """
    data = np.loadtxt(data_file)
    fs = 4096
    window = 'hann'
    n = 256
    f, t, Z = stft(data, fs=fs, window=window, nperseg=n)
    plt.figure(figsize=(4, 4))
    plt.pcolormesh(t, f, np.abs(Z),shading='auto', cmap='jet')
    # 关闭坐标轴
    plt.axis('off')
    mid_dir = data_file.split('\\')[-2] + '_' + data_file.split('\\')[-1].split('.')[0]
    # 保存图像
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    savefile = save_root + '\\' + mid_dir
    plt.savefig(savefile, bbox_inches='tight', pad_inches=0,dpi=200)
    plt.close('all')



def generate_img(data_csv_file,save_root):
    """
    读取保存每个样本的路径和标签的csv文件，并根据样本路径，将其转换为相应图片后保存在相应的路径下
    :param data_csv_file: csv的路径值
    :param save_root: 保存目录
    :return:
    """
    data_csv = pd.read_csv(data_csv_file)
    data_file = data_csv['data'].values.tolist()
    data_label = data_csv['label'].values.tolist()
    for i in range(len(data_file)):
        img_file = data_file[i]
        img_label = data_label[i]
        data2cwt(img_file,save_root+f'\cwt\{img_label}')
        data2stft(img_file, save_root + f'\stft\{img_label}')

def generate_unlabel_img(data_files,save_root):
    """
    读取保存每个样本的路径和标签的csv文件，并根据样本路径，将其转换为相应图片后保存在相应的路径下
    :param data_csv_file: csv的路径值
    :param save_root: 保存目录
    :return:
    """
    data_file = os.listdir(data_files)
    for i in range(len(data_file)):
        img_file = os.path.join(data_files,data_file[i])
        data2cwt(img_file,save_root+'\cwt')
        data2stft(img_file, save_root + '\stft')

if __name__=='__main__':
    #原始训练集中没有标签的数据集地址
    unlabeled_data_files = r'D:\Study\competition_lerning\YiDongComp\Data_version\training\unlabeled'
    #生成的没标签的训练集图片保存的文件夹
    unlabeled_save_root = r'D:\Study\competition_lerning\Yidongbei2\Data_version\training\img_unlabeled'
    generate_unlabel_img(unlabeled_data_files,unlabeled_save_root)
    # Step1生成的训练集的csv文件地址
    train_csv_file = r'D:\Study\competition_lerning\Yidongbei2\trian.csv'
    #生成的有标签的训练集图片的保存文件夹
    train_save_root = r'D:\Study\competition_lerning\Yidongbei2\Data_version\training\img_labeled'
    generate_img(train_csv_file,train_save_root)