import random
import numpy as np
import pandas as pd
from libsvm.svmutil import *

class A9a():
    def __init__(self, n_clients=3, train_root='./data/a9a/a9a', test_root='./data/a9a/a9a.t'):
        random.seed(0)
        np.random.seed(0)
        y_train, x_train = svm_read_problem(train_root)
        y_test, x_test = svm_read_problem(test_root)
        # print('[A9a] train_samples:', len(y_train), 'test_samples:', len(y_test))

        all_feature_train = []
        for i in range(len(y_train)):
            all_feature_train.append([0.0 for _ in range(123)])
            for item in x_train[i].items(): all_feature_train[i][item[0]-1] = 1.0
            if y_train[i]<0: y_train[i]=0
            else: y_train[i]=1
        all_feature_test = []
        for i in range(len(y_test)):
            all_feature_test.append([0.0 for _ in range(123)])
            for item in x_test[i].items(): all_feature_test[i][item[0]-1] = 1.0
            if y_test[i]<0: y_test[i]=0
            else: y_test[i]=1

        self.all_feature_train = np.array(all_feature_train)
        self.all_feature_test = np.array(all_feature_test)
        self.label_train = np.array(y_train)
        self.label_test = np.array(y_test)
        # print('train:', self.all_feature_train.shape, self.label_train.shape)
        # print('test:', self.all_feature_test.shape, self.label_test.shape)

        self.n_clients = n_clients
    
    def get_train(self):
        size = self.all_feature_train.shape[1] // self.n_clients
        features = []
        for i in range(self.n_clients):
            if i == self.n_clients - 1: features.append(self.all_feature_train[:, i*size:])
            else: features.append(self.all_feature_train[:, i*size:(i+1)*size])
        labels = [self.label_train]
        return features, labels
    
    def get_train_merged(self):
        all_feature = self.all_feature_train
        labels = [self.label_train]
        return all_feature, labels
    
    def get_test(self):
        size = self.all_feature_test.shape[1] // self.n_clients
        features = []
        for i in range(self.n_clients):
            if i == self.n_clients - 1: features.append(self.all_feature_test[:, i*size:])
            else: features.append(self.all_feature_test[:, i*size:(i+1)*size])
        labels = [self.label_test]
        return features, labels
