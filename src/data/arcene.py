import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
class Arcene:

    def __init__(self):
        self.train_dir =  'src/dataset/arcene/arcene_train.data'
        self.train_labels_dir = 'src/dataset/arcene/arcene_train.labels'
        self.test_dir = 'src/dataset/arcene/arcene_valid.data'
        self.test_labels_dir = 'src/dataset/arcene/arcene_valid.labels'

        pass

    def load_data(self):
        train_data = pd.read_csv(self.train_dir, sep=' ', header=None)
        train_labels = pd.read_csv(self.train_labels_dir, sep=' ', header=None)
        train_data.dropna(axis=1,how='all',inplace=True)
        
        test_data = pd.read_csv(self.test_dir, sep=' ', header=None)
        test_labels = pd.read_csv(self.test_labels_dir, sep=' ', header=None)
        test_data.dropna(axis=1,how='all',inplace=True)
        #test_labels = pd.read_csv(self.test_labels_dir, sep=' ', header=None)


        return train_data, train_labels, test_data, test_labels

    def get_data(self):
        train_data, train_labels, test_data, test_labels = self.load_data()
        X = pd.concat([train_data, test_data],axis=0)
        Y = pd.concat([train_labels, test_labels], axis=0)
        X=X.values
        Y=Y.values.ravel()

        scaler = MinMaxScaler()

        scaler.fit(X)

        X = scaler.transform(X)

        for i in range(X.shape[1]):
        #print(i,np.mean(X[:,i]),np.std(X[:,i]))
            if np.std(X[:,i])==0:
                X[:,i]+=np.random.normal(0,0.001,X.shape[0])


        return X,Y


