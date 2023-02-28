# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年09月01日
"""
import numpy as np # linear algebra
import pandas as pd
import os

def create_traincsv(data_dir,train_save):
    path_files = []
    labels = []
    for parent, _,files in os.walk(data_dir):
        for file in files:
            path_files.append( os.path.join(parent,file))
            labels.append(parent.split('\\')[-1])
    dic = {"data":path_files,'label':labels}
    dic = pd.DataFrame(dic)
    dic.to_csv(train_save,index=False)

if __name__=='__main__':
    #训练集所在的文件夹
    train_dir = r'D:\Study\competition_lerning\YiDongComp\Data_version\training\labeled'
    #输出的csv文件保存地址
    train_save = r'D:\Study\competition_lerning\Yidongbei2\trian.csv'
    create_traincsv(train_dir,train_save)