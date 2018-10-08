# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:18:17 2018

@author: xiaolala
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

class GA_TOP(object):
    def __init__(self,class_size,feature_size,pop_size,pc,pm,iteration,code_pool_size,
                 train_x,train_y,validate_x,validate_y,test_x,test_y):
        self.class_size = class_size
        self.feature_size = feature_size
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.iteration = iteration
        self.code_pool_size = code_pool_size
        self.train_x = train_x
        self.train_y = train_y
        self.validate_x = validate_x
        self.validate_y = validate_y
        self.test_x = test_x
        self.test_y = test_y
    
    '''generate first one of two code_matrix[numpy] and fs_matrix[list]'''
    def generateCode(self):
        code_matrix = []
        
        for i in range(self.class_size):
            temp = []
            for j in range(self.num_classifier):
                temp.append(random.randint(-1,1))
            code_matrix.append(temp)
        code_matrix = np.array(code_matrix)
        
        '''#随机的特征选择矩阵
        fs_matrix = []
        for i in range(num_classifier):
            temp=[]
            for j in range(self.feature_size):
                temp.append(random.randint(0,1))
            fs_matrix.append(temp)
        new_matrix = []
        for line in fs_matrix:
            temp = []
            for j in range(len(line)):
                if line[j]==1:
                    temp.append(j)
            new_matrix.append(temp)
        fs_matrix = new_matrix'''
        
        #全部考虑的特征选择矩阵
        fs_matrix = []
        for i in range(self.num_classifier):
            fs_matrix.append(range(self.feature_size))
        
        self.legalityExamination(code_matrix,fs_matrix)
        return code_matrix,fs_matrix
    
    '''generate first ternary operation line[list]'''
    def generateTOP(self,num_classifier):
        top_lines = []
        for i in range(self.pop_size):
            top_line = []
            for j in range(num_classifier):
                # 1-5代表5种三进制运算:加减乘(除)与或
                temp_column = []
                # 第一个矩阵的第几列
                temp_column.append(random.randint(0,self.code_pool_size-1))
                temp_column.append(random.randint(0,self.num_classifier-1))
                # 第二个矩阵的第几列
                temp_column.append(random.randint(0,self.code_pool_size-1))
                temp_column.append(random.randint(0,self.num_classifier-1))
                # 操作符，1-5为5种三进制运算：加减乘(除)与或，6为置零，7/8为取一半，9/10为取奇偶
                temp_column.append(random.randint(1,10))
                top_line.append(temp_column)
            top_lines.append(top_line)
        return top_lines
    
    '''generate sub_code_matrix[list[numpy]] with code_matrix operating through top'''
    def generateSubCodes(self,code_matrixs,top_lines):
        new_code_matrixs = []
        for top_line in top_lines:
            new_code_matrix = []
            for i in range(len(top_line)):
                temp_code = []
                # 三进制运算
                if top_line[i][-1] in range(1,6):
                    for j in range(self.class_size):
                        temp_code.append(self.topCalculate(code_matrixs[top_line[i][0]][j][top_line[i][1]],code_matrixs[top_line[i][2]][j][top_line[i][3]],top_line[i][4]))
                # 全置为0，null
                elif top_line[i][-1]==6:
                    for j in range(self.class_size):
                        temp_code.append(0)
                # 第一个矩阵某列前半部分 + 第二个矩阵某列后半部分
                elif top_line[i][-1]==7:
                    for j in range(round(self.class_size/2)):
                        temp_code.append(code_matrixs[top_line[i][0]][j][top_line[i][1]])
                    for j in range(round(self.class_size/2),self.class_size):
                        temp_code.append(code_matrixs[top_line[i][2]][j][top_line[i][3]])
                # 第一个矩阵某列后半部分 + 第二个矩阵某列前半部分
                elif top_line[i][-1]==8:
                    for j in range(round(self.class_size/2)):
                        temp_code.append(code_matrixs[top_line[i][2]][j][top_line[i][3]])
                    for j in range(round(self.class_size/2),self.class_size):
                        temp_code.append(code_matrixs[top_line[i][0]][j][top_line[i][1]])
                # 第一个矩阵某列奇数行 + 第二个矩阵某列偶数行
                elif top_line[i][-1]==9:
                    for j in range(self.class_size):
                        if j%2==0:
                            temp_code.append(code_matrixs[top_line[i][2]][j][top_line[i][3]])
                        else:
                            temp_code.append(code_matrixs[top_line[i][0]][j][top_line[i][1]])
                # 第一个矩阵某列偶数行 + 第二个矩阵某列奇数行
                elif top_line[i][-1]==10:
                    for j in range(self.class_size):
                        if j%2==0:
                            temp_code.append(code_matrixs[top_line[i][0]][j][top_line[i][1]])
                        else:
                            temp_code.append(code_matrixs[top_line[i][2]][j][top_line[i][3]])
                new_code_matrix.append(temp_code)
                '''若全为0，直接去掉'''
                if top_line[i][-1]==6:
                    del new_code_matrix[-1]
            new_code_matrix = np.transpose(new_code_matrix)
            #self.legalityExamination(new_code_matrix,self.fs_matrix)
            new_code_matrix = self.greedyExamination(new_code_matrix,self.fs_matrix)
            new_code_matrixs.append(new_code_matrix)
        return new_code_matrixs
                    
    
    '''seven types tenary operating rules'''
    def topCalculate(self,a,b,operation):
        if operation==1:
            if a==-1:
                if b==-1 or b==0:
                    return -1
                elif b==1:
                    return 0
            elif a==0:
                return b
            elif a==1:
                if b==-1:
                    return 0
                elif b==0 or b==1:
                    return 1
        elif operation==2:
            if a==-1:
                if b==-1:
                    return 0
                elif b==0 or b==1:
                    return -1
            elif a==0:
                return -b
            elif a==1:
                if b==-1 or b==0:
                    return 1
                elif b==1:
                    return 0
        elif operation==3:
            if a==-1:
                return -b
            elif a==0:
                return 0
            elif a==1:
                return b
        elif operation==4:
            if a==-1:
                return b
            elif a==0:
                return 0
            elif a==1:
                if b==-1 or b==1:
                    return 1
                elif b==0:
                    return 0
        elif operation==5:
            if a==-1:
                if b==-1 or b==1:
                    return -1
                elif b==0:
                    return 0
            elif a==0:
                return 0
            elif a==1:
                return b
        
    '''calculate the scores of current code_matrix'''
    def calValue(self,code_matrix,fs_matrix,dataType):
        estimator = KNeighborsClassifier(n_neighbors=3)
        #estimator = DecisionTreeClassifier()
        #estimator = SVC()
        #code_matrix = [[-1,1,-1],[0,-1,-1],[1,-1,1]]
        ecoc_classifier = ECOCClassifier2(estimator, code_matrix.tolist(), fs_matrix)
        if dataType=="validate":
            predict_y = ecoc_classifier.fit_predict(self.train_x, self.train_y, self.validate_x)
            accuracy = ms.f1_score(self.validate_y,predict_y,average="micro")
        elif dataType=="test":
            predict_y = ecoc_classifier.fit_predict(self.train_x, self.train_y, self.test_x)
            accuracy = ms.f1_score(self.test_y,predict_y,average="micro")
        return accuracy
    
    '''sort'''
    def sort(self,top_lines,top_code_matrixs,top_values):
        for i in range(len(top_values)-1):
            for j in range(len(top_values)-i-1):
                if(top_values[j]<top_values[j+1]):
                    temp = top_values[j]
                    top_values[j] = top_values[j+1]
                    top_values[j+1] = temp
                    temp = top_lines[j]
                    top_lines[j] = top_lines[j+1]
                    top_lines[j+1] = temp
                    temp = top_code_matrixs[j]
                    top_code_matrixs[j] = top_code_matrixs[j+1]
                    top_code_matrixs[j+1] = temp
    
    '''generate the index list of cross/mutation'''
    def generateIndex(self,p,num_classifier):
        Count = round(num_classifier*p)
        if Count==0:
            Count = 1
        counter = 0
        index = []
        while(counter<Count):
            tempInt = random.randint(0,num_classifier-1)
            if tempInt not in index:
                index.append(tempInt)
                counter = counter+1
        return index
    
    '''cross over'''
    def cross(self,top_lines,num_classifier):
        temp_lines = copy.deepcopy(top_lines)
        for i in range(1,len(temp_lines)):
            index = self.generateIndex(self.pc,num_classifier)
            for j in index:
                #temp_lines[i-1][j],temp_lines[i][j] = temp_lines[i][j],temp_lines[i-1],[j]
                temp = temp_lines[i-1][j]
                temp_lines[i-1][j] = temp_lines[i][j]
                temp_lines[i][j] = temp
        return temp_lines
    
    '''mutation'''
    def mutation(self,top_lines,num_classifier):
        for i in range(len(top_lines)):
            index = self.generateIndex(self.pm,num_classifier)
            for j in index:
                index_in_column = self.generateIndex(random.randint(1,3)/5,5)
                for k in index_in_column:
                    if k==0 or k==2:
                        tempInt = random.randint(0,self.code_pool_size-1)
                        while tempInt==top_lines[i][j][k]:
                            tempInt = random.randint(0,self.code_pool_size-1)
                    elif k==1 or k==3:
                        tempInt = random.randint(0,self.num_classifier-1)
                        while tempInt==top_lines[i][j][k]:
                            tempInt = random.randint(0,self.num_classifier-1)
                    elif k==4:
                        tempInt = random.randint(1,10)
                        while tempInt==top_lines[i][j][k]:
                            tempInt = random.randint(1,10)
                    top_lines[i][j][k] = tempInt
    
    '''贪心合法性检查'''
    def greedyExamination(self,code_matrix,fs_matrix):
        flag = False
        index = 0
        while index<code_matrix.shape[1]:
            # 每一列必须包含1和-1
            # 若该列全为1或全为-1
            if (code_matrix[:,index]==1).all()==True or (code_matrix[:,index]==-1).all()==True:
                flag = True
                # 如果列超过1.5*class_size则删除该列
                if code_matrix.shape[1]>=round(self.class_size*1.5):
                    code_matrix = np.delete(code_matrix,index,axis=1)
                    continue
                # 列未超过1.5*class_size则取前2/n个取反
                else:
                    bound = round(code_matrix.shape[0]/2)
                    if bound==0:
                        bound=1
                    for j in range(bound+1):
                        if j%2==0:
                            code_matrix[j][index] = 0
                        code_matrix[j][index] = -1*code_matrix[j][index]
            # 若全为0
            elif (code_matrix[:,index]==0).all()==True:
                flag=True
                if code_matrix.shape[1]>=round(self.class_size*1.5):
                    code_matrix = np.delete(code_matrix,index,axis=1)
                    continue
                else:
                    for j in range(code_matrix.shape[0]):
                        if j%2==0:
                            code_matrix[j][index]=1
                        else:
                            code_matrix[j][index]=-1
            # 全为0和-1
            elif (code_matrix[:,index]==1).any()==False:
                flag=True
                column = code_matrix[:,index].tolist()
                position = column.index(0)
                code_matrix[position,index]=1
            # 全为0和1
            elif (code_matrix[:,index]==-1).any()==False:
                flag=True
                column = code_matrix[:,index].tolist()
                position = column.index(0)
                code_matrix[position,index]=-1
            # 不能含有相同或相反的列
            temparray = np.zeros(self.class_size)
            transpose_code_matrix = np.transpose(code_matrix)
            temp_code_matrix = np.delete(transpose_code_matrix,index,axis=0)
            flag2=False
            for j in range(temp_code_matrix.shape[0]):
                if ((transpose_code_matrix[index]-temp_code_matrix[j])==temparray).all() or ((transpose_code_matrix[index]+temp_code_matrix[j])==temparray).all():
                    flag = True
                    # 如果列超出1.5*class_size则删除该列
                    if code_matrix.shape[1]>=round(self.class_size*1.5):
                        code_matrix = np.delete(code_matrix,index,axis=1)
                        flag2=True
                        break
                    # 如果列未超出则取前2/n个取反
                    else:
                        bound = round(code_matrix.shape[0]/2)
                        if bound==0:
                            bound=1
                        for j in range(bound+1):
                            if code_matrix[j][index]==1:
                                code_matrix[j][index]=0
                            elif code_matrix[j][index]==0:
                                code_matrix[j][index]=-1
                            elif code_matrix[j][index]==-1:
                                code_matrix[j][index]=1
            if flag2==True:
                continue
            index = index + 1
        i=0
        temparray = np.zeros(code_matrix.shape[1])
        for line in code_matrix:
            #不能含有全为0的行
            if (line==temparray).all():
                print("66666")
                print(line)
                flag = True
                bound = round(code_matrix.shape[1]/2)
#                if bound==0:
#                    bound=1
                for j in range(bound+1):
                    if j%2==0:
                        line[j]=1
                    else:
                        line[j]=-1
            #不能含有相同的行
            temp_code_matrix = np.delete(code_matrix,i,axis=0)
            for j in range(temp_code_matrix.shape[0]):
                if ((line-temp_code_matrix[j])==temparray).all():
                    print("77777")
                    print(line)
                    flag = True
                    bound = round(code_matrix.shape[1]/2)
                    if bound==0 and len(line)>1:
                        bound=1
                    for j in range(bound+1):
                        if line[j]==1:
                            line[j]=0
                        elif line[j]==0:
                            line[j]=-1
                        elif line[j]==-1:
                            line[j]=1
            i=i+1
        if flag==True:
            self.greedyExamination(code_matrix,fs_matrix)
        return code_matrix
    
    '''examination of legality突变合法性检查'''
    def legalityExamination(self,code_matrix,fs_matrix):
        i=0
        flag = False
        temparray = np.zeros(code_matrix.shape[1])
        for line in code_matrix:
            #不能含有全为0的行
            while (line==temparray).all():
                index = random.randint(0,code_matrix.shape[1]-1)
                line[index] = random.randint(-1,1)
                flag=True
            #不能含有相同的行
            temp_code_matrix = np.delete(code_matrix,i,axis=0)
            for j in range(temp_code_matrix.shape[0]):
                while ((line-temp_code_matrix[j])==temparray).all():
                    index = random.randint(0,code_matrix.shape[1]-1)
                    line[index] = random.randint(-1,1)
                    flag=True
            i=i+1
        
        temparray = np.zeros(self.class_size)
        for i in range(code_matrix.shape[1]):
            #每一列必须包含1和-1
            while (code_matrix[:,i]==1).any()==False or (code_matrix[:,i]==-1).any()==False:
                index = random.randint(0,self.class_size-1)
                code_matrix[index][i] = random.randint(-1,1)
                flag=True
            #不能含有相同或相反的列
            transpose_code_matrix = np.transpose(code_matrix)
            temp_code_matrix = np.delete(transpose_code_matrix,i,axis=0)
            for j in range(temp_code_matrix.shape[0]):
                while ((transpose_code_matrix[i]-temp_code_matrix[j])==temparray).all() or ((transpose_code_matrix[i]+temp_code_matrix[j])==temparray).all():
                    index = random.randint(0,self.class_size-1)
                    transpose_code_matrix[i][index] = random.randint(-1,1)
                    code_matrix[index][i] = transpose_code_matrix[i][index]
                    flag=True
        if flag==True:
            self.legalityExamination(code_matrix,fs_matrix)
                
    
    '''main'''
    def main(self):
        doc = open('C:/Users/24320/Desktop/output.txt','w')
        
        code_matrixs = []
        self.num_classifier = self.class_size*2
        fs_matrixs = []
        
        #self.num_classifier = random.randint(self.class_size-1,round(self.class_size*2))
        self.num_classifier = self.class_size*2
        print("生成初始矩阵:",file=doc)
        for i in range(self.code_pool_size):
            code_matrix,fs_matrix = self.generateCode()
            code_matrixs.append(code_matrix)
            print(i,file=doc)
            print(code_matrix,file=doc)
            fs_matrixs.append(fs_matrix)
        
        
        print("生成初始三进制运算字符串top_lines:",file=doc)
        top_lines = self.generateTOP(self.num_classifier)
        print(top_lines,file=doc)
        
        fs_matrix = []
        for i in range(self.num_classifier):
            temp = []
            for j in range(self.feature_size):
                temp.append(random.randint(0,1))
            fs_matrix.append(temp)
        self.fs_matrix = fs_matrix
        
        accuracy = []   #best accuracy of each ietration
        test_accuracy = []  #best result for test data
        avg_accuracy = []
        avg_test_accuracy = []
        
        '''begin GA'''
        for i in range(self.iteration):
            print("第"+str(i)+"代",file=doc)
            print("top_lines:",file=doc)
            print(top_lines,file=doc)
            print("生成top_lines对应的子矩阵:",file=doc)
            top_code_matrixs = self.generateSubCodes(code_matrixs,top_lines)
            print(top_code_matrixs,file=doc)
            top_values = []
            for j in range(len(top_code_matrixs)):
                top_values.append(self.calValue(top_code_matrixs[j],fs_matrix,"validate"))
            print("top_lines对应的子矩阵分数:")
            print(top_values)
            
            '''sort top lines according to top_values'''
            self.sort(top_lines,top_code_matrixs,top_values)
            print("排序:",file=doc)
            for j in range(len(top_values)):
                print(top_code_matrixs[j],file=doc)
                print(top_values[j],file=doc)
            accuracy.append(max(top_values))
            avg_accuracy.append(sum(accuracy)/len(accuracy))
            
            print("最大值:")
            print(max(top_values))
            print(top_code_matrixs[0])
            
            test_accuracy.append(self.calValue(top_code_matrixs[0],fs_matrix,"test"))
            avg_test_accuracy.append(sum(test_accuracy)/len(test_accuracy))
            
            temp_top_lines = copy.deepcopy(top_lines)
            random.shuffle(temp_top_lines)
            son_top_lines = self.cross(temp_top_lines,self.num_classifier)
            print("交叉:",file=doc)
            print(son_top_lines,file=doc)
            self.mutation(son_top_lines,self.num_classifier)
            print("变异:",file=doc)
            print(son_top_lines,file=doc)
            son_code_matrixs = self.generateSubCodes(code_matrixs,son_top_lines)
            print("新生成的子矩阵:",file=doc)
            print(son_code_matrixs,file=doc)
            son_values = []
            for j in range(len(son_code_matrixs)):
                son_values.append(self.calValue(son_code_matrixs[j],fs_matrix,"validate"))
            self.sort(son_top_lines,son_code_matrixs,son_values)
            print("新生成的子矩阵分数:")
            print(son_values)
            print(son_top_lines)
            if top_values[0]>=son_values[-1]:
                print("精英保留:")
                print(top_lines[0])
                print(top_values[0])
                son_values[-1] = top_values[0]
                son_top_lines[-1] = top_lines[0]
            top_lines = son_top_lines
            random.shuffle(top_lines)
        
        doc.close()
        
        return accuracy,test_accuracy,avg_accuracy,avg_test_accuracy
        
'''
'abalone','cmc','dermatology',
            '''
datasets = ['ecoli','glass','iris','optdigits'
            ,'sat','thyroid','vertebral']
for dataset in datasets:
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
        class_size = len(np.unique(np.array(train_y)))
        feature_size = len(train_x[0])
        pop_size = 25   #种群个体数量
        pc = 0.4
        pm = 0.2
        iteration = 60
        code_pool_size = 15
        ga_top = GA_TOP(class_size,feature_size,pop_size,pc,pm,iteration,code_pool_size,
                        train_x,train_y,validate_x,validate_y,test_x,test_y)
        
        
        accuracy,test_accuracy,avg_accuracy,avg_test_accuracy = ga_top.main()  
        '''print out the result'''
        plt.plot(accuracy,label = "Validate")
        #plt.plot(avg_accuracy,label = "Va",color = "g")
        plt.plot(test_accuracy,label="Test",color="r")
        #plt.plot(avg_test_accuracy,label="Ta",color="m")
        plt.legend()
        
        
        filename = "./figures/" + str(dataset) + "/" + str(dataset) + str(i+1) + ".png"
        plt.savefig(filename)
        plt.show()
        
        recordfilename = "./record/"  + str(dataset) + ".txt"
        recordfile =open(recordfilename,"a")
        print(i,file=recordfile)
        print("ga_top最大值:",file=recordfile)
        print(max(test_accuracy),file=recordfile)
        print("ga_top平均值:",file=recordfile)
        print(sum(test_accuracy)/len(test_accuracy),file=recordfile)
        recordfile.close()
    