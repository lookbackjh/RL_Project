# this is a code for testing the RL based Feature Selection method
from src.data.twomoon import Twomoon_synthetic
from src.model.rl import FeatureSelection
from src.model.new_try import FeatureSelection_Backward
import argparse
from src.data.datawrapper import DataWrapper
import numpy as np
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description='RL based Feature Selection')
#parser.add_argument('--seed', type=int, default=12345, help='random seed')

parser.add_argument('--episode_number', type=int, default=10000, help='number of episodes')
parser.add_argument('--alpha', type=float, default=0.1, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.3, help='discount factor')
parser.add_argument('--epsilon', type=float, default=0.3, help='epsilon greedy')
parser.add_argument('--predefined_reward', type=bool, default=False, help='predefined reward')
parser.add_argument('--correlation_loss_coefficient', type=float, default=0.001, help='correlation loss coefficient')
parser.add_argument('--backward', type=bool, default=True, help='backward elimination')
parser.add_argument('--backward_tau', type=float, default=0.0001, help='threshold tau for backward elimination')
parser.add_argument('--datatype', type=str, default="spambase", help='arcene twomoon, spambase available')
parser.add_argument('--worsening_count', type=int, default=3, help='worsening count') 
parser.add_argument('--is_custom', type=bool, default=True, help='custom reward')
args = parser.parse_args()



if __name__ == '__main__':
    # create a dataset
    #twomoon = Twomoon_synthetic(100,1000)
    
    # get x,y 
    X,Y=DataWrapper(args).get_data()
    # traintest split
    #train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.3)
    if args.datatype== "arcene" :
        args.worsening_count=5


    
    # traintest split
    #train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.3)
    

    rl=FeatureSelection_Backward(args,X,Y)
    aorvalues=rl.run()
    print((aorvalues))
    print(np.argsort(aorvalues)[::-1])
    print(rl.feature_counts)
    print("Backward Elimination")

    import json

    a={}
    strr=args.datatype+"_alpha_"+str(args.alpha)+"_gamma_"+str(args.gamma)+"_eps_"+str(args.epsilon)+"_backward_"+str(args.backward)+"_tau_"+str(args.backward_tau)+"_cor_"+str(args.correlation_loss_coefficient)+"_custom_"+str(args.is_custom)
    str_json=strr+".json"
    with open(str_json, 'w') as fp:
        a["aorvalues"]=aorvalues.tolist()
        a["feature_order"]=np.argsort(aorvalues)[::-1].tolist()
        a["feature_counts"]=rl.feature_counts.tolist()
        json.dump(a,fp)


    
    # print(train_x.shape) # sample size x feature size
    # print(train_y)
