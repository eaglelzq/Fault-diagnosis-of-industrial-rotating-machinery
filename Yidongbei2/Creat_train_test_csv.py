# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年09月01日
"""
import numpy as np # linear algebra
import pandas as pd

import os




def creat_img_train(data_dir_img,save_root):
    """
        生成用于保存每个训练集的图片地址的csv文件
    :param data_dir_img: 上一步生成的带标签的训练集图片地址
    :param save_root: csv文件的存放地址
    :return:
    """
    path_files = []
    labels = []
    for parent, _,files in os.walk(data_dir_img):
        for file in files:
            path_files.append(os.path.join(parent,file))
            labels.append(parent.split('\\')[-1])
    dic = {"data":path_files,'label':labels}
    dic = pd.DataFrame(dic)
    dic.to_csv(save_root,index=False)


def re_dir(str_list,flag):
    if flag:
        path = 'D:\\'
        for i in str_list[1:]:
            path = os.path.join(path,i)
    else:
        path = ''
        for i in str_list[:]:
            path = os.path.join(path, i)
    return path
def creat_img_train_test(data_dir_img,save_root):
    path_files = []
    labels = []
    cwt_dir = data_dir_img+'\cwt'
    for parent, _,files in os.walk(cwt_dir):
        for file in files:
            path_files.append(os.path.join(parent,file))
            labels.append(parent.split('\\')[-1])
    dic = {"cwt_data":path_files,'label':labels}
    dic = pd.DataFrame(dic)
    dic['stft_data'] = dic['cwt_data'].apply(lambda x : os.path.join(data_dir_img,'stft',re_dir(x.split('\\')[-2:],False)))
    # dic['gray_data'] = dic['cwt_data'].apply(lambda x: os.path.join(data_dir_img, 'gray', re_dir(x.split('\\')[-2:],False)))
    # dic['GAF_data'] = dic['cwt_data'].apply(lambda x: os.path.join(data_dir_img, 'GAF', re_dir(x.split('\\')[-2:], False)))

    dic.to_csv(save_root,index=False)


if __name__=='__main__':
    #step2生成训练集图片的文件夹地址
    train_img_dir = r'D:\Study\competition_lerning\鹰鹰鹰_工业旋转机械设备健康状态检测V4_15310257053\Yidongbei2\Data_version\training\img_labeled'
    #生成的训练集csv文件的保存地址
    train_save_dir = r'D:\Study\competition_lerning\鹰鹰鹰_工业旋转机械设备健康状态检测V4_15310257053\Yidongbei2\Data_version\training\total_train.csv'
    creat_img_train_test(train_img_dir,train_save_dir)