# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:57:10 2018

@author: xiaolala
"""

import DataLoader
from GA_TOP import *
import numpy as np

if __name__=="__main__":
    ##########################################################################
    #DataLoader
    ##########################################################################
    trainfile = "./data_uci/abalone_train.data"
    testfile = "./data_uci/abalone_test.data"
    validatefile = "./data_uci/abalone_validation.data"
    # 其中x为特征空间，y为样本的标签
    train_x, train_y, validate_x, validate_y,instance_size = DataLoader.loadDataset(trainfile,validatefile)
    train_x, train_y, test_x, test_y, instance_size = DataLoader.loadDataset(trainfile, testfile)
    
    ##########################################################################
    # 配置参数运行程序
    ##########################################################################
    class_size = len(np.unique(np.array(train_y)))
    feature_size = len(train_x[0])
    pop_size = 10   #种群个体数量
    pc = 0.4
    pm = 0.1
    iteration = 30
    ga_top = GA_TOP(class_size,feature_size,pop_size,pc,pm,iteration,
                 train_x,train_y,validate_x,validate_y,test_x,test_y)
    ga_top.main()