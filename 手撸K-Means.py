'''
    K-Means
'''
import numpy as np
from random import randint

class K_Means():
    def __init__(self, K = None, init_K = None):
        if K == None:
            self.K = 2
        else:
            self.K = K
        self.init_K = init_K
        self.First = True
        self.K_final = None
        self.Mean = None
        self.Std = None

    def zScoreNormalization(self, X):
        '''
            标准化
        :param X:
        :return:
        '''
        if self.First:
            self.Mean, self.Std = list(), list()

        for index in range(self.shape[-1]):
            if self.First:
                self.Mean.append(np.mean(self.X[:,index]))
                self.Std.append(np.std(self.X[:,index]))
            X[:, index] = (X[:, index] - self.Mean[index]) / self.Std[index]
        return X

    def euclidDistance(self, list1, list2):
        '''
            欧氏距离度量
        :return:
        '''
        sum = 0
        for num1, num2 in zip(list1, list2):
            sum += (num1 - num2) ** 2
        return sum ** (1/2)

    def predict(self,X):
        '''
            预测分类
        :param X:
        :return:
        '''
        self.Label = list()
        for Data_index, line in enumerate(X):
            euclidTemp = None
            minIndex = 0
            for K_index in range(self.K):
                euclidValue = self.euclidDistance(self.init_K[K_index], line)
                if euclidTemp == None:
                    euclidTemp = euclidValue
                elif euclidValue < euclidTemp:
                    minIndex = K_index
            self.Label.append(minIndex)
        return self.Label

    def transformK(self,label = None):
        '''
            更新质点
        :param label:
        :return:
        '''
        if self.First:
            self.init_K = dict()
        K_Temp = dict()
        K_count = dict()
        if label != None:
            for index in range(self.K):
                K_Temp[index] = np.array([0 for _ in range(self.shape[-1])],np.float)
                K_count[index] = 0
            for index, line in enumerate(self.X):
                K_Temp[label[index]] += line
                K_count[label[index]] += 1
            for index in range(self.K):
                if K_count[index] != 0:
                    K_Temp[index] = K_Temp[index] / K_count[index]
        for i in range(self.K):
            if self.First:
                self.init_K[i] = list()

            for j in range(self.shape[-1]):
                if self.First:
                    K_temp = randint(int(self.shapeMin[j]),int(self.shapeMax[j])) / 1000
                else:
                    K_temp = K_Temp[i][j]
                if self.First:
                    self.init_K[i].append(K_temp)
                else:
                    self.init_K[i][j] = K_temp
        self.First = False
        return self.init_K

    def fit(self,X):
        self.X = np.array(X,np.float)
        self.shape = np.shape(self.X)

        '''
            标准化数据
            初始化质点

        '''
        self.X = self.zScoreNormalization(self.X)
        self.shapeMax, self.shapeMin = dict(), dict()
        for i in range(self.shape[-1]):
            self.shapeMax[i] = max(self.X[:,i]) * 1000
            self.shapeMin[i] = min(self.X[:,i]) * 1000
        self.transformK()
        '''
            计算K点距离

        '''
        i = 0
        while self.K_final != self.transformK(self.predict(self.X)):
            i += 1
            print("迭代量 ： ", i)
            self.K_final = self.init_K
        return True

data = list()
for i in range(1024):
    data.append(list())
    for j in range(2):
        data[i].append(randint(-5894,4561))

model = K_Means(2)
model.fit(data)
print(model.Label)
print(model.X)
