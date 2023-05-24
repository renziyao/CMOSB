import random
import numpy as np
import pandas as pd
import sklearn.datasets

class Synthetic():
    def __init__(self, n_samples=1000, n_features=5, n_clients=3, n_classes=2, n_clusters=5):
        self.data = sklearn.datasets.make_classification(
            n_samples=n_samples, 
            n_features=n_features * n_clients, 
            n_redundant=0,
            n_informative=n_features * n_clients, 
            n_classes=n_classes, 
            n_clusters_per_class=n_clusters, 
            shuffle=True, 
            random_state=0,
        )
        n = self.data[0].shape[0] * 4 // 5
        all_feature = self.data[0][:n]
        it = 0
        features = []
        for i in range(n_clients):
            features.append(all_feature[:, it: it + n_features])
            it += n_features
        labels = [self.data[1][:n]]
        self.train_features = features
        self.train_labels = labels
        n = self.data[0].shape[0] * 1 // 5
        all_feature = self.data[0][-n:]
        it = 0
        features = []
        for i in range(n_clients):
            features.append(all_feature[:, it: it + n_features])
            it += n_features
        labels = [self.data[1][-n:]]
        self.test_features = features
        self.test_labels = labels
    
    def get_train(self):
        return self.train_features, self.train_labels
    
    def get_test(self):
        return self.test_features, self.test_labels

