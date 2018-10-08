# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 08:37:14 2018

@author: 24320
"""
import numpy as np
import random
import copy
import sklearn.metrics as ms
import matplotlib.pyplot as plt
import CodeMatrix.CodeMatrix as CM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from ECOCClassifier import SimpleECOCClassifier
from BaseClassifier import get_base_clf
from ECOCClassfier import ECOCClassifier2
import DataLoader
from sklearn.svm import SVC

datasets = ['abalone','cmc','dermatology','ecoli','glass','iris','optdigits'
            ,'sat','thyroid','vertebral']
for dataset in datasets:
    test_accuracy = []
    for i in range(10):
        ##########################################################################
        #DataLoader
        ##########################################################################
        trainfile = "./data_uci/" + str(dataset) + "_train.data"
        testfile = "./data_uci/" + str(dataset) + "_test.data"
        validatefile = "./data_uci/" + str(dataset) + "_validation.data"
        # 其中x为特征空间，y为样本的标签
        train_x, train_y, validate_x, validate_y,instance_size = DataLoader.loadDataset(trainfile,validatefile)
        train_x, train_y, test_x, test_y, instance_size = DataLoader.loadDataset(trainfile, testfile)
            
        ##########################################################################
        # 配置参数运行程序
        ##########################################################################
        code_matrix,index = CM.sparse_rand(train_x,train_y)
        print(CM.ova(train_x,train_y))
        print(CM.dense_rand(train_x,train_y))
        print(CM.sparse_rand(train_x,train_y))
        estimator = SVC()
        
        sec = SimpleECOCClassifier(estimator, code_matrix)
        sec.fit(train_x, train_y)
        pred = sec.predict(test_x)
        accuracy = ms.f1_score(test_y,pred,average="micro")
        test_accuracy.append(accuracy)
        
        
    recordfilename = "./record2/"  + str(dataset) + ".txt"
    recordfile =open(recordfilename,"a")
    print(i,file=recordfile)
    print("baseline最大值:",file=recordfile)
    print(max(test_accuracy),file=recordfile)
    print("baseline平均值:",file=recordfile)
    print(min(test_accuracy),file=recordfile)
    recordfile.close()