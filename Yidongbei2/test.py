# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年09月05日
"""
import numpy as np
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
data = np.random.randn(5)
data2 = np.random.randint(1,10,4)
print(astype(data,data.dtype))
