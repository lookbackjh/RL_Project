import numpy as np
from sklearn.svm import SVC 
from collections import defaultdict
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
class FeatureSelection():

    def __init__(self, args,train_x,train_y ,test_x,test_y):
        self.args = args
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x  
        self.test_y=test_y
        self.feature_size=train_x.shape[1]
        self.aormean=np.zeros(self.feature_size)
        self.cur_value=0
        self.worsening_count=0
        self.aorcount=np.zeros(self.feature_size)

    def reward(self,cur_feature:tuple):
    
        # cur_x to be elementwise multiplication of train_x and cur_feature with is binary
        
        # cur feature is tuple of indexes, so need to convert it to array
        if len(cur_feature)==0:
            return 0


        cur_feature=list(cur_feature)

        cur_x=self.train_x[:,cur_feature]
        rf=SVC()
        rf.fit(cur_x,self.train_y)
        cur_test_x=self.test_x[:,cur_feature]

        return rf.score(cur_test_x, self.test_y )

    def exploit_based(self,cur_feature:tuple):
        
        # select feature taht maximizes and action a in node F 
        cur_feature_frozen=frozenset(cur_feature)

        T_f=self.V_counter[cur_feature_frozen] # denotes the number of times that the feature subset F has been selected
        b=0.7

        if round(T_f**b)==round((T_f+1)**b):

            actions=self.V_successor[cur_feature_frozen] # denotes the actions that have been taken in node F that is, actions denote new states that is derived from F
            t_f_actions=[self.V_A_counter[(cur_feature_frozen,frozenset(action))] for action in (actions)] # denotes the number of times that each action has been taken in node F
            
            mu_f_actions=[self.Value[action] for action in (actions)] # denotes the AOR values of each action in node F

            a_hat=np.argmax(mu_f_actions+np.sqrt(2*np.log(T_f)/t_f_actions))

            # update V_A_counter
            #self.V_A_counter[(cur_feature_frozen,actions[a_hat])]+=1

            return actions[a_hat]
    
        else:

            # select the feature that has not been selected before that has maximum AOR value
            # if there are multiple features that have the same AOR value, then select the feature that has been selected the least number of times
            aoridxs=np.argsort(self.aormean)[::-1]
            for idx in aoridxs:
                if idx not in cur_feature:
                    new_feature=idx
                    break
            
            new_feature= cur_feature+(new_feature,)

            if new_feature not in self.V_successor[cur_feature_frozen]:
                self.V_successor[cur_feature_frozen].append((new_feature))

            return new_feature


    def update(self, cur_feature:tuple, next_feature:tuple,previous_value:float):

        # update Values of current state, successor state, and AOR values
        frozen_cur_feature=frozenset(cur_feature)   
        frozen_next_feature=frozenset(next_feature)
        self.Value[frozen_cur_feature]+=self.args.alpha*(self.reward(frozen_next_feature)-self.reward(frozen_cur_feature)+self.args.gamma*self.Value[frozen_next_feature]-self.Value[frozen_cur_feature])


        # Update AOR 
        selected_feature=frozen_next_feature.difference(frozen_cur_feature)
        selected_feature=list(selected_feature)[0]

        self.aorcount[selected_feature]+=1
        k=self.aorcount[selected_feature]

        rewarddiff=self.reward(frozen_next_feature)-self.reward(frozen_cur_feature)
        self.aormean[selected_feature]=(rewarddiff+(k-1)*self.aormean[selected_feature])/k
        # Update V_counter
        self.V_counter[frozen_next_feature]+=1


        pass

    def explore_based(self,cur_feature:tuple):
        
        # select the feature that has not been selected before. uniformly
        feature_candidates=[i for i in range(self.feature_size) if i not in cur_feature]
        
        new_feature=np.random.choice(feature_candidates)
        next_state=cur_feature+(new_feature,)
        cur_feature_frozen=frozenset(cur_feature)
        #next_state_frozen=frozenset(next_state)
        self.V_successor[cur_feature_frozen].append(next_state)

        # cur feature is tuple,  and i want to return tuple with new feature aded
        return next_state



    def stop_condition(self,cur_feature:tuple, next_feature:tuple,previous_value:float):
        # if the reward is not improving, stop the episode
        if previous_value<self.Value[cur_feature]:
            self.worsening_count=0
            pass
        else:
            self.worsening_count+=1

        if self.worsening_count>=5:
            self.worsening_count=0
            return True
        else:
            return False

    def run(self):
        '''
            (1) The main algorithm of FSTD: Since the method is iterative, initially the number of iterations is given as input; the algorithm
        sets the AOR values and visitation numbers to zero for all features and initializes the graph in the zero level. At the beginning
        of each iteration, it starts from an empty state and calls a function to select a feature.
        If the current state has been seen in the graph and some features have been selected here, t 
        '''
        feature_tuples=[]
        self.AOR_values=defaultdict(list)

        # I want to make dictionary of tuple where each of the tuple indicates the index of the features.
        self.V_counter=defaultdict(float)
        self.Value=defaultdict(float)

        # This is the dictionary of the successor of each state.
        self.V_successor=defaultdict(list)
        self.V_A_counter=defaultdict(float)

        

        graph=[]
        for i in tqdm(range(self.args.episode_number)):
            
            cur_feature=()   
            # need to define each state.  is there any way to  define array-> value dictionary or set?
            #self.V_counter[frozenset(cur_feature)]+=1
            while True:
                 # should return an array of features

                if frozenset(cur_feature)  in graph:
                    # if the next feature is in the graph, then we do the exploitation chosse feature with largest aors\
                    # epsilon greedy
                    if np.random.rand()<0.05:
                        next_feature=self.explore_based(cur_feature)
                    else:
                        next_feature=self.exploit_based(cur_feature)
                else:  
                    next_feature=self.explore_based(cur_feature)
                    if cur_feature !=():
                        graph.append(frozenset(cur_feature))


                pair=(frozenset(cur_feature),frozenset(next_feature))
                self.V_A_counter[pair]+=1

                previous_value=self.Value[frozenset(cur_feature)]
                self.update(cur_feature,next_feature,previous_value)


                if self.stop_condition(cur_feature, next_feature,previous_value):
                    self.worsening_count=0
                    break
                
                
                
                
                
                cur_feature=next_feature
                
                


        return self.aormean



