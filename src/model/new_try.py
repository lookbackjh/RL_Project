import numpy as np
from sklearn.svm import SVC 
from collections import defaultdict
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
class FeatureSelection_Backward():

    def __init__(self, args,data_X,data_Y):
        self.args = args
        self.data_X=data_X
        self.data_Y=data_Y
        self.feature_size=data_X.shape[1]
        self.aormean=np.zeros(self.feature_size)
        self.cur_value=0
        self.worsening_count=0
        self.aorcount=np.zeros(self.feature_size)
        # feature_size* feature_size matrix
        self.correlation_=np.corrcoef(data_X,rowvar=False)
        self.feature_counts=np.zeros(self.feature_size)
        self.random_state=12345


    def custom_reward(self,cur_feature:tuple, next_feature:tuple):
    
        # cur feature is tuple of indexes, so need to convert it to array
        cur_feature=list(cur_feature)
        next_feature=list(next_feature)

        train_X,test_X,train_Y,test_Y=train_test_split(self.data_X,self.data_Y,test_size=0.4,random_state=self.random_state)
        

        if self.args.datatype=="spambase":
            train_X,test_X,train_Y,test_Y=train_test_split(self.data_X,self.data_Y,test_size=0.3,random_state=self.random_state)
            train_X=train_X[:500,:]
            train_Y=train_Y[:500]
            test_X=test_X[:300,:]
            test_Y=test_Y[:300]

        
        svc_cur=SVC(C=5.0, kernel='rbf')
        svc_cur.fit(train_X[:,cur_feature],train_Y)
        cur_test_x=test_X[:,cur_feature]
        cur_reward=svc_cur.score(cur_test_x, test_Y )

        svc_next=SVC(C=5.0, kernel='rbf')
        svc_next.fit(train_X[:,next_feature],train_Y)
        next_test_x=test_X[:,next_feature]
        next_reward=svc_next.score(next_test_x, test_Y )

        # feature number with largest aorvalue is selected
        cur_best=np.argsort(self.aormean)[::-1]
        new_feature= np.setdiff1d(cur_feature,next_feature)[0]

        total_reward=-next_reward+cur_reward +self.args.correlation_loss_coefficient*self.correlation_[cur_best,new_feature]

        return total_reward


    def reward_diff(self,cur_feature:tuple,next_feature:tuple):
        
        # cur feature is tuple of indexes, so need to convert it to array


        

        train_X,test_X,train_Y,test_Y=train_test_split(self.data_X,self.data_Y,test_size=0.4,random_state=self.random_state)
        

        if self.args.datatype=="spambase":
            train_X,test_X,train_Y,test_Y=train_test_split(self.data_X,self.data_Y,test_size=0.3,random_state=self.random_state)
            train_X=train_X[:500,:]
            train_Y=train_Y[:500]
            test_X=test_X[:300,:]
            test_Y=test_Y[:300]

        
        svc_cur=SVC(C=5.0, kernel='rbf')
        svc_cur.fit(train_X[:,cur_feature],train_Y)
        cur_test_x=test_X[:,cur_feature]
        cur_reward=svc_cur.score(cur_test_x, test_Y )

        svc_next=SVC(C=5.0, kernel='rbf')
        svc_next.fit(train_X[:,next_feature],train_Y)
        next_test_x=test_X[:,next_feature]
        next_reward=svc_next.score(next_test_x, test_Y )

        total_reward=next_reward-cur_reward

        return total_reward
    

    def stop_condition(self,cur_feature:tuple, next_feature:tuple,previous_value:float):
        # if the reward is not improving, stop the episode
        diff=self.reward_diff(cur_feature,next_feature)
        #print(abs(diff))
        if abs(diff)>self.args.backward_tau:
            self.worsening_count+=1
            pass
        else:
            pass
            

        if self.worsening_count>=self.args.worsening_count:
            self.worsening_count=0
            return True
        else:
            return False
    
    def reward(self,cur_feature:tuple):
    
        # cur_x to be elementwise multiplication of train_x and cur_feature with is binary
        
        # cur feature is tuple of indexes, so need to convert it to array
        # if len(cur_feature)==0:
        #     return 0


        cur_feature=list(cur_feature)



        train_X,test_X,train_Y,test_Y=train_test_split(self.data_X,self.data_Y,test_size=0.4,random_state=self.random_state)



        
        if self.args.datatype=="spambase":
            train_X,test_X,train_Y,test_Y=train_test_split(self.data_X,self.data_Y,test_size=0.3,random_state=self.random_state)
            train_X=train_X[:500,:]
            train_Y=train_Y[:500]
            test_X=test_X[:300,:]
            test_Y=test_Y[:300]
        cur_x=train_X[:,cur_feature]
        cur_test_x=test_X[:,cur_feature]

        #lgb=LGBMClassifier(verbose=-1)
        lgb=SVC(C=5.0,kernel='rbf')
        lgb.fit(cur_x,train_Y)
        gbmscore=lgb.score(cur_test_x, test_Y )

        #lgb.fit(cur_x,train_Y)
        
        #self.feature_counts[cur_feature]+=1


        #rf.score(cur_test_x, test_Y )

        return -gbmscore

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
            cur_feature_list=list(cur_feature)
            removed_feature=np.setdiff1d(np.arange(self.feature_size),cur_feature_list)   ## 이미 지워진건 생각을 안하도록. 

            aoridxs=np.argsort(self.aormean)[::-1]
            for idx in aoridxs:
                if idx not in removed_feature:
                    new_feature=idx
                    break
            

            new_feature_list=[i for i in cur_feature_list if i!=new_feature]
            new_feature_tuple=tuple(new_feature_list)



            if new_feature_tuple not in (self.V_successor[cur_feature_frozen]):
                self.V_successor[cur_feature_frozen].append((new_feature_tuple))



            return new_feature_tuple


    def update(self, cur_feature:tuple, next_feature:tuple,previous_value:float):

        # update Values of current state, successor state, and AOR values
        frozen_cur_feature=frozenset(cur_feature)   
        frozen_next_feature=frozenset(next_feature)


        if self.args.is_custom:
            self.Value[frozen_cur_feature]+=self.args.alpha*(self.custom_reward(frozen_cur_feature,frozen_next_feature)+self.args.gamma*self.Value[frozen_next_feature]-self.Value[frozen_cur_feature])
        else:
            self.Value[frozen_cur_feature]+=self.args.alpha*(self.reward(frozen_next_feature)-self.reward(frozen_cur_feature)+self.args.gamma*self.Value[frozen_next_feature]-self.Value[frozen_cur_feature])


        # Update AOR 
        selected_feature=frozen_cur_feature.difference(frozen_next_feature)
        selected_feature=list(selected_feature)[0]

        self.aorcount[selected_feature]+=1
        k=self.aorcount[selected_feature]

        self.feature_counts[selected_feature]+=1

    

        rewarddiff=self.reward(frozen_next_feature)-self.reward(frozen_cur_feature)
        self.aormean[selected_feature]=(rewarddiff+(k-1)*self.aormean[selected_feature])/k
        # Update V_counter
        self.V_counter[frozen_next_feature]+=1
        self.random_state+=1

        pass

    def explore_based(self,cur_feature:tuple):
        
        # select the feature that has not been selected before. uniformly

        cur_feature_list=list(cur_feature)
        removed_feature=np.setdiff1d(np.arange(self.feature_size),cur_feature_list)

        feature_candidates=[i for i in range(self.feature_size) if i not in removed_feature]
        new_feature_to_remove=np.random.choice(feature_candidates)
        
        # else:
        #     new_feature=np.random.choice(feature_candidates)
        
    
        next_state=[i for i in cur_feature_list if i!=new_feature_to_remove]
        cur_feature=tuple(cur_feature_list)
        cur_feature_frozen=frozenset(cur_feature)
        #next_state_frozen=frozenset(next_state)
        next_state=tuple(next_state)
        self.V_successor[cur_feature_frozen].append(next_state)

        # cur feature is tuple,  and i want to return tuple with new feature aded
        return next_state


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
            

            cur_feature=np.arange(self.feature_size)
            # randomly select half of the feature
            # np.random.shuffle(cur_feature)

            # if self.args.datatype=="arcene":
            #     np.random.shuffle(cur_feature)
            #     #cur_feature=cur_feature[:300]
                

            # else:
            #     cur_feature=cur_feature[:int(self.feature_size)]
            cur_feature=tuple(cur_feature)

            # need to define each state.  is there any way to  define array-> value dictionary or set?
            #self.V_counter[frozenset(cur_feature)]+=1
            while True:
                 # should return an array of features

                if len(cur_feature)==1:
                    self.worsening_count=0
                    break
                if frozenset(cur_feature)  in graph:
                    # if the next feature is in the graph, then we do the exploitation chosse feature with largest aors\
                    # epsilon greedy
                    next_feature=self.exploit_based(cur_feature)
                else:  
                    next_feature=self.explore_based(cur_feature)

                graph.append(frozenset(cur_feature))

                pair=(frozenset(cur_feature),frozenset(next_feature))
                self.V_A_counter[pair]+=1

                previous_value=self.Value[frozenset(cur_feature)]
                self.update(cur_feature,next_feature,previous_value)


                

                if self.stop_condition(cur_feature,next_feature,previous_value):
                    self.worsening_count=0
                    break


                cur_feature=next_feature
            
            
            print(cur_feature)
            print(self.reward(cur_feature))
            print(len(cur_feature))    
            
                

        return self.aormean



