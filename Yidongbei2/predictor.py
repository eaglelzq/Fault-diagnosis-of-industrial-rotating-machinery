"""
作者:Eagle
日期:2022年07月26日
"""
from data_precoss import load_data,data_processor
import torch
import os
from DAN import DANNet
import numpy as np


class predictor:
    def __init__(self,test_path,model_path):
        self.test_dir = os.path.join(test_path,'samples')
        self.save_dir = model_path
    def predict(self):
        test_files = load_data(self.test_dir)
        result = []
        model_file = os.path.join(self.save_dir,'my_model.params')
        model = DANNet()
        model.load_state_dict(torch.load(model_file))
        model.eval()
        for i in test_files:
            in_data = data_processor(i)
            res = model.predict(in_data)
            result.append(np.argmax(res.detach().numpy(),axis=1)[0])
        np.savetxt(os.path.join('.', 'preLabel.txt'), np.array(result), fmt='%d')
if __name__=='__main__':
    #原始测试集数据路径
    test_file = r'D:\Study\competition_lerning\YiDongComp\Data_version\testing'
    #模型保存路径不用更改
    save_path = os.path.join('.','model_saver')
    pre = predictor(test_file,save_path)
    pre.predict()