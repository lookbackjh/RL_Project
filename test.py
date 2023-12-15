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
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon greedy')
parser.add_argument('--predefined_reward', type=bool, default=False, help='predefined reward')
parser.add_argument('--correlation_loss_coefficient', type=float, default=0.01, help='correlation loss coefficient')
parser.add_argument('--datatype', type=str, default="arcene", help='arcene twomoon, spambase available')
parser.add_argument('--worsening_count', type=int, default=5, help='worsening count')   
args = parser.parse_args()



if __name__ == '__main__':
    # create a dataset
    X,Y=DataWrapper(args).get_data()
    # traintest split
    #train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.3)
    if args.datatype== "arcene" :
        args.worsening_count=10

    rl=FeatureSelection(args,X,Y)
    aorvalues=rl.run()
    print((aorvalues))
    print(np.argsort(aorvalues)[::-1])
    print(rl.feature_counts)
    print(args.predefined_reward)
    
    # save as json

    import json
    pre=[False]

    for p in pre:
        args.predefined_reward=p
        strr=args.datatype+"_predef_"+str(args.predefined_reward)+"_alpha_"+str(args.alpha)+"_gamma_"+str(args.gamma)+"_eps_"+str(args.epsilon)+"_cor_"+str(args.correlation_loss_coefficient)
        str_json=strr+".json"


        with open(str_json, 'w') as fp:
            json.dump("aorvalues\n",fp)
            fp.write('\n')
            json.dump(aorvalues.tolist(), fp)
            fp.write('\n')
            json.dump("feature order\n",fp)
            fp.write('\n')
            json.dump(np.argsort(aorvalues)[::-1].tolist(), fp)
            fp.write('\n')
            json.dump("feature counts\n",fp)
            fp.write('\n')
            json.dump(rl.feature_counts.tolist(), fp)
            fp.write('\n')
            json.dump("predefined_reward\n",fp)
            fp.write('\n')
            json.dump(args.predefined_reward, fp)
            fp.write('\n')
            json.dump("datatype\n",fp)
            fp.write('\n')
            json.dump(args.datatype, fp)
            fp.write('\n')

    
    # print(train_x.shape) # sample size x feature size
    # print(train_y)
