import pandas as pd
class Spam:

    def __init__(self):
        self.train_dir = 'src/dataset/spambase' + '/spambase.data'

        pass

    def load_data(self):
        X = pd.read_csv(self.train_dir, sep=',', header=None)
        #data.dropna(axis=1, how='all', inplace=True)
        X=X.values
        Y=X[:,-1]
        X=X[:,:-1]
        return X,Y