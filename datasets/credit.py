import random
import numpy as np
import pandas as pd

class Credit():
    def __init__(self, n_clients=3, root='./data/credit/cs-training.csv'):
        random.seed(0)
        np.random.seed(0)
        data = pd.read_csv(root)
        data = data.fillna(data.mean())
        # data_1 = data[data['SeriousDlqin2yrs'] == 1].iloc[:10000]
        # data_0 = data[data['SeriousDlqin2yrs'] == 0].iloc[:10000]
        # data = pd.concat([data_0, data_1])
        data = data.sample(frac=1)
        self.data = data
        self.n_clients = n_clients
    
    def get_train(self):
        n = self.data.shape[0] * 2 // 3
        all_feature = self.data.iloc[:n,2:].to_numpy()
        size = all_feature.shape[1] // self.n_clients
        features = []
        for i in range(self.n_clients):
            if i == self.n_clients - 1: features.append(all_feature[:, i*size:])
            else: features.append(all_feature[:, i*size:(i+1)*size])
        # features = [
        #     all_feature[:, 0:4],
        #     all_feature[:, 4:7],
        #     all_feature[:, 7:10],
        # ]
        labels = [self.data.iloc[:n, 1].to_numpy()]
        return features, labels
    
    def get_train_merged(self):
        n = self.data.shape[0] * 2 // 3
        all_feature = self.data.iloc[:n,2:].to_numpy()
        labels = [self.data.iloc[:n, 1].to_numpy()]
        return all_feature, labels
    
    def get_test(self):
        n = self.data.shape[0] * 1 // 3
        all_feature = self.data.iloc[-n:,2:].to_numpy()
        size = all_feature.shape[1] // self.n_clients
        features = []
        for i in range(self.n_clients):
            if i == self.n_clients - 1: features.append(all_feature[:, i*size:])
            else: features.append(all_feature[:, i*size:(i+1)*size])
        # features = [
        #     all_feature[:, 0:4],
        #     all_feature[:, 4:7],
        #     all_feature[:, 7:10],
        # ]
        labels = [self.data.iloc[-n:, 1].to_numpy()]
        return features, labels
