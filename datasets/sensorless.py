import random
import numpy as np
import pandas as pd

class Sensorless():
    def __init__(self, n_clients=3, root='./data/sensorless/Sensorless_drive_diagnosis.txt'):
        assert n_clients==2
        random.seed(0)
        np.random.seed(0)
        data = pd.read_csv(root, delimiter=' ', header=None)
        data = data.fillna(data.mean())
        data = data.sample(frac=1.0)
        self.data = data
        self.n_clients = n_clients
    
    def get_train(self):
        n = self.data.shape[0] * 2 // 3
        all_feature = self.data.iloc[:n,:48].to_numpy()
        size = all_feature.shape[1] // self.n_clients
        features = []
        # for i in range(self.n_clients):
        #     if i == self.n_clients - 1: features.append(all_feature[:, i*size:])
        #     else: features.append(all_feature[:, i*size:(i+1)*size])
        features.append(all_feature[:, :12])
        features.append(all_feature[:, 12:48])
        labels = [self.data.iloc[:n, -1].to_numpy()-1]
        return features, labels
    
    def get_train_merged(self):
        n = self.data.shape[0] * 2 // 3
        all_feature = self.data.iloc[:n,:48].to_numpy()
        labels = [self.data.iloc[:n, -1].to_numpy()-1]
        return all_feature, labels
    
    def get_test(self):
        n = self.data.shape[0] * 1 // 3
        all_feature = self.data.iloc[-n:,:48].to_numpy()
        size = all_feature.shape[1] // self.n_clients
        features = []
        # for i in range(self.n_clients):
        #     if i == self.n_clients - 1: features.append(all_feature[:, i*size:])
        #     else: features.append(all_feature[:, i*size:(i+1)*size])
        features.append(all_feature[:, :12])
        features.append(all_feature[:, 12:48])
        labels = [self.data.iloc[-n:, -1].to_numpy()-1]
        return features, labels

if __name__ == '__main__':
    sensorless = Sensorless()
    print(sensorless.data.shape)
    all_feature, labels = sensorless.get_train_merged()
    print(labels[0].min(), labels[0].max())
