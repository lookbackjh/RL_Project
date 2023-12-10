# this is a code for testing the RL based Feature Selection method
from src.data.twomoon import Twomoon_synthetic
from src.model.rl import FeatureSelection
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description='RL based Feature Selection')
#parser.add_argument('--seed', type=int, default=12345, help='random seed')

parser.add_argument('--episode_number', type=int, default=50000, help='number of episodes')
parser.add_argument('--alpha', type=float, default=5, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon greedy')
parser.add_argument('--predefined_reward', type=bool, default=True, help='predefined reward')
parser.add_argument('--correlation_loss_coefficient', type=float, default=0.01, help='correlation loss coefficient')
args = parser.parse_args()



if __name__ == '__main__':
    # create a dataset
    twomoon = Twomoon_synthetic(100,1000)
    
    # get x,y 
    X,Y=twomoon.create_data()
    # traintest split
    #train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.3)
    

    rl=FeatureSelection(args,X,Y)
    aorvalues=rl.run()
    print((aorvalues))
    print(np.argsort(aorvalues)[::-1])
    
    # print(train_x.shape) # sample size x feature size
    # print(train_y)
