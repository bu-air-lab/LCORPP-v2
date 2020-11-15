
from pathlib import Path
from utils.parser import Policy,Solver
from utils.pomdp_parser import Model
from utils.pomdp_solver import generate_policy
import subprocess
import numpy as np
import random

class Planner:
    def __init__(self, solve_pomdp=False):          
        solver_path = Path.cwd()/'model/pomdpsol'
		#assert Path(pomdp_file).is_file(), 'POMDP path does not exist'

        pomdp_file=Path.cwd()/'model/program.pomdp'
        assert Path(pomdp_file).is_file(), 'POMDP path does not exist'
            
        self.model = Model(pomdp_file=pomdp_file, parsing_print_flag=False)
        #print(self.model.states,self.model.actions)
	
        policy_file = Path.cwd()/'model/program.policy'
        assert Path(policy_file).is_file(), 'POMDP path does not exist'
        #print (policy_file)
        print(self.model.states,self.model.actions)
            
        if solve_pomdp:
            subprocess.run(["./pomdpsol program.pomdp --timeout 20 --output program.policy"],cwd='model')   
			#generate_policy(solver_path,pomdp_file,policy_file)
            
        self.policy = Policy(len(self.model.states),
							len(self.model.actions),
							policy_file=policy_file)
      
    def initialize_belief(self,p_co,p_em):
        p_co=round(float(p_co),4)
        p_em=round(float(p_em),4)
        b1=0.5*p_co*p_em
        b2=0.5*p_co*(1-p_em)
        b3=0.5*(1-p_co)*p_em
        b4=0.5*(1-p_co)*(1-p_em)
        b5=0.5*p_co*p_em
        b6=0.5*p_co*(1-p_em)
        b7=0.5*(1-p_co)*p_em
        b8=0.5*(1-p_co)*(1-p_em)
        belief=np.array([b1,b2,b3,b4,b5,b6,b7,b8,0.0])          
        #belief=np.
        #belief=np.array([0.5*p_co,0.5*(1-p_co),0.5*p_co,0.5*(1-p_co),0.0])
        return belief
      
    def initialize_state_p_distribution(self,p_co,p_em,p_room):
        p_co=round(float(p_co),4)
        p_em=round(float(p_em),4)
        p_room=round(float(p_room),4)
        b1=p_room*p_co*p_em
        b2=p_room*p_co*(1-p_em)
        b3=p_room*(1-p_co)*p_em
        b4=p_room*(1-p_co)*(1-p_em)
        b5=(1-p_room)*p_co*p_em
        b6=(1-p_room)*p_co*(1-p_em)
        b7=(1-p_room)*(1-p_co)*p_em
        b8=(1-p_room)*(1-p_co)*(1-p_em)
        belief=np.array([b1,b2,b3,b4,b5,b6,b7,b8,0.0])          
        #belief=np.
        #belief=np.array([0.5*p_co,0.5*(1-p_co),0.5*p_co,0.5*(1-p_co),0.0])
        return belief
      
    def update_belief(self,a_idx,o_idx,belief):
        belief=np.dot(belief,self.model.trans_mat[a_idx, :])
        belief=[belief[i] * self.model.obs_mat[a_idx, i, o_idx] for i in range(len(self.model.states))]
        belief=belief/sum(belief)
        return belief
      	#def update_policy
      
    def observe(self,a_idx,next_state):
        s_idx = self.model.states.index(next_state)
        return np.random.choice(self.model.observations, p= self.model.obs_mat[a_idx,s_idx,:])
      
    def run(self,instance):
        p_co=instance['co_reasoning']
        p_em=instance['cr_reasoning']
        belief=self.initialize_belief(p_co,p_em)
        #gt corresponding to ground truth
        if instance['co']=="cooperative":
            gt_co=1.0
        elif instance['co']=='not cooperative':
            gt_co=0.0
            
        if instance['cr']=="empty":
            gt_em=1.0
        elif instance['cr']=='crowded':
            gt_em=0.0
            
        if instance['room_left']=="room":
            gt_room=1.0
        elif instance['room_left']=='no room':
            gt_room=0.0
            
        gt_belief=self.initialize_state_p_distribution(gt_co,gt_em,gt_room)
        #print("\nGt distribution is",gt_belief)
        print("The initial belief is:", belief)
        instance["belief"]=belief.tolist()
        instance['pomdp']=[]
        state= np.random.choice(self.model.states[:-1],p=gt_belief[:-1])
        # make the initial state agree with the initial belief
        r=0
        term=False
        while not term:
            a_idx=self.policy.select_action(belief)
            s_idx = self.model.states.index(state)
            r=r+self.model.reward_mat[a_idx,s_idx]
                  
            print ('\nUnderlying state: ', state)
            instance['pomdp'].append("Underlying state:"+state)
                  
            print ('action is: ',self.model.actions[a_idx])
            instance['pomdp'].append("Action:"+self.model.actions[a_idx])
            #print("actions are",self.model.actions)
            next_state = np.random.choice(self.model.states, p=self.model.trans_mat[a_idx,s_idx,:])
                  
            obs = self.observe(a_idx,next_state)
            print("obs is:",obs)
            instance['pomdp'].append("Obs:"+obs)
                  
            obs_idx = self.model.observations.index(obs)
            #print ('observation is: ',self.model.observations[obs_idx])
            #instance['pomdp'].append("Obs:"+self.model.observations[obs_idx])
                  
            belief=self.update_belief(a_idx,obs_idx,belief)
            instance['pomdp'].append(belief.tolist())
            print(belief)
                  
            if belief[-1]>0:
                term=True
                instance['cost']=(r-self.model.reward_mat[a_idx,s_idx])*(-1)
                instance['reward']=r                    
                if(state=="R_W_E" or state=="R_W_E_not"):
                    #print("Successful")
                    return "Successful"
                else:
                    return "Failed"
                    #print("\n")
            state=next_state
            
    def run_pr(self,instance):
        p_co=instance['co_reasoning']
        p_em=instance['cr_reasoning']
        belief=self.initialize_belief(p_co,p_em)
            
        #gt corresponding to ground truth
        if instance['co']=="cooperative":
            gt_co=1
        elif instance['co']=='not cooperative':
            gt_co=0

        if instance['cr']=="empty":
            gt_em=1.0
        elif instance['cr']=='crowded':
            gt_em=0.0
            
        if instance['room_left']=="room":
            gt_room=1.0
        elif instance['room_left']=='no room':
            gt_room=0.0
            
        gt_belief=self.initialize_state_p_distribution(gt_co,gt_em,gt_room)
        print("The initial belief is:", belief)
        instance["belief"]=belief.tolist()
        instance['pomdp']=[]
        state= np.random.choice(self.model.states[:-1],p=gt_belief[:-1])
        # make the initial state agree with the initial belief
        r=0
        term=False
            
        #do fixed sequential actions
        #S0,A0 is "left signal"
        s_idx = self.model.states.index(state)
        a_idx=0
        print("state is: ",self.model.states[s_idx])
        print ('action is: ',self.model.actions[a_idx])
        next_state = np.random.choice(self.model.states, p=self.model.trans_mat[a_idx,s_idx,:])
        obs = self.observe(a_idx,next_state)
            
        #R1     
        r=r+self.model.reward_mat[a_idx,s_idx]
        instance['pomdp'].append("Action:"+self.model.actions[a_idx])
        obs_idx = self.model.observations.index(obs)
        belief=self.update_belief(a_idx,obs_idx,belief)

        while belief[0]<0.7 and belief[1]<0.7:         
            #S1,A1 is "move left"
            s_idx = self.model.states.index(next_state)
            #a_idx=random.choice([0,1])
            a_idx=0
            print ('action is: ',self.model.actions[a_idx])
            #R2
            r=r+self.model.reward_mat[a_idx,s_idx]
            instance['pomdp'].append("Action:"+self.model.actions[a_idx])
            next_state = np.random.choice(self.model.states, p=self.model.trans_mat[a_idx,s_idx,:])
            obs = self.observe(a_idx,next_state)
            obs_idx = self.model.observations.index(obs)
            belief=self.update_belief(a_idx,obs_idx,belief)            
            #S2,A2 is "merge left"
            #next_state = np.random.choice(self.model.states, p=self.model.trans_mat[a_idx,s_idx,:])
            
            s_idx = self.model.states.index(next_state)
            a_idx=2
            print ('action is: ',self.model.actions[a_idx])
            
            #R3
            r=r+self.model.reward_mat[a_idx,s_idx]
            instance['pomdp'].append("Action:"+self.model.actions[a_idx])

            #S3
            #state=np.random.choice(self.model.states, p=self.model.trans_mat[a_idx,s_idx,:])
            instance['cost']=(r-self.model.reward_mat[a_idx,s_idx])*(-1)
            instance['reward']=r                    
            if(next_state=="R_W_E" or next_state=="R_W_E_not"):
            #print("Successful")
                return "Successful"
            else:
                return "Failed"