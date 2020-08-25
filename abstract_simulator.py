# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:02:17 2020

@author: cckklt
"""
import argparse
import json
import random
import numpy as np
import statistics as st
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess
from utils.parser import Policy,Solver
from utils.pomdp_parser import Model
from utils.pomdp_solver import generate_policy
np.set_printoptions(precision=2)     # for better belief printing 

random.seed(5)
np.random.seed(5)

class Reasoner:
      @staticmethod
      def minor_output_evd(f,instance,index):
            
            if instance['weather']=="sunny":
                  f.write("Weather(Sunny,"+index+")\n")
                  f.write("!Weather(Rainy,"+index+")\n")
            elif instance['weather']=="rainy":
                  f.write("!Weather(Sunny,"+index+")\n")
                  f.write("Weather(Rainy,"+index+")\n")
      
            if instance['time_period']=="busy":
                  f.write("Time(Busy,"+index+")\n")
                  f.write("!Time(Normal,"+index+")\n")
            elif instance['time_period']=="normal":
                  f.write("Time(Normal,"+index+")\n")
                  f.write("!Time(Busy,"+index+")\n")
            
            if instance['perception']=="crowded":
                  f.write("Perception(Crowded,"+index+")\n")
                  f.write("!Perception(Empty,"+index+")\n")
            
            elif instance['perception']=="empty":
                  f.write("!Perception(Crowded,"+index+")\n")
                  f.write("Perception(Empty,"+index+")\n")

      def minor_output_data(f,instance,index):
            if instance['weather']=="sunny":
                  f.write("Weather(Sunny,"+index+")\n")
                  f.write("!Weather(Rainy,"+index+")\n")
            elif instance['weather']=="rainy":
                  f.write("!Weather(Sunny,"+index+")\n")
                  f.write("Weather(Rainy,"+index+")\n")
      
            if instance['time_period']=="busy":
                  f.write("Time(Busy,"+index+")\n")
                  f.write("!Time(Normal,"+index+")\n")
            elif instance['time_period']=="normal":
                  f.write("Time(Normal,"+index+")\n")
                  f.write("!Time(Busy,"+index+")\n")
                        
            if instance['cr']=="crowded":
                  f.write("Road(Crowded,"+index+")\n")
                  f.write("!Road(Empty,"+index+")\n")
            
            elif instance['cr']=="empty":
                  f.write("!Road(Crowded,"+index+")\n")
                  f.write("Road(Empty,"+index+")\n")
                     
            if instance['perception']=="crowded":
                  f.write("Perception(Crowded,"+index+")\n")
                  f.write("!Perception(Empty,"+index+")\n")
            
            elif instance['perception']=="empty":
                  f.write("!Perception(Crowded,"+index+")\n")
                  f.write("Perception(Empty,"+index+")\n")
            
            if instance['co']=="cooperative":
                  f.write("Cooperative("+index+")\n")
            elif instance['co']=="not cooperative": 
                  f.write("!Cooperative("+index+")\n")
            
      
      @staticmethod
      def output_evidence(f,instance,index):
            #f=open(f)    
            if instance['weather']=="sunny":
                  f.write("Weather(Sunny,"+index+")\n")
                  f.write("!Weather(Rainy,"+index+")\n")
            elif instance['weather']=="rainy":
                  f.write("!Weather(Sunny,"+index+")\n")
                  f.write("Weather(Rainy,"+index+")\n")
      
            if instance['time_period']=="busy":
                  f.write("Time(Busy,"+index+")\n")
                  f.write("!Time(Normal,"+index+")\n")
            elif instance['time_period']=="normal":
                  f.write("Time(Normal,"+index+")\n")
                  f.write("!Time(Busy,"+index+")\n")
      
            if instance['vehicle_type']=="sedan":
                  f.write("Vehicle(Car,"+index+")\n")
                  f.write("!Vehicle(Sport,"+index+")\n")
                  f.write("!Vehicle(Truck,"+index+")\n")
      
            elif instance['vehicle_type']=="sport":
                  f.write("Vehicle(Sport,"+index+")\n")
                  f.write("!Vehicle(Car,"+index+")\n")
                  f.write("!Vehicle(Truck,"+index+")\n")
               
            elif instance['vehicle_type']=="truck":
                  f.write("Vehicle(Truck,"+index+")\n")
                  f.write("!Vehicle(Car,"+index+")\n")
                  f.write("!Vehicle(Sport,"+index+")\n")
            
            if instance['perception']=="crowded":
                  f.write("Perception(Crowded,"+index+")\n")
                  f.write("!Perception(Empty,"+index+")\n")
            
            elif instance['perception']=="empty":
                  f.write("!Perception(Crowded,"+index+")\n")
                  f.write("Perception(Empty,"+index+")\n")
            
            f.write("\n")
            #f.close()
      
      @staticmethod
      def output_data(data_file,instance,index):
            #f=open(f)    
            f=open("reasoner/"+data_file,"a")
            if instance['weather']=="sunny":
                  f.write("Weather(Sunny,"+index+")\n")
                  f.write("!Weather(Rainy,"+index+")\n")
            elif instance['weather']=="rainy":
                  f.write("!Weather(Sunny,"+index+")\n")
                  f.write("Weather(Rainy,"+index+")\n")
      
            if instance['time_period']=="busy":
                  f.write("Time(Busy,"+index+")\n")
                  f.write("!Time(Normal,"+index+")\n")
            elif instance['time_period']=="normal":
                  f.write("Time(Normal,"+index+")\n")
                  f.write("!Time(Busy,"+index+")\n")
      
            if instance['vehicle_type']=="sedan":
                  f.write("Vehicle(Car,"+index+")\n")
                  f.write("!Vehicle(Sport,"+index+")\n")
                  f.write("!Vehicle(Truck,"+index+")\n")
      
            elif instance['vehicle_type']=="sport":
                  f.write("Vehicle(Sport,"+index+")\n")
                  f.write("!Vehicle(Car,"+index+")\n")
                  f.write("!Vehicle(Truck,"+index+")\n")
               
            elif instance['vehicle_type']=="truck":
                  f.write("Vehicle(Truck,"+index+")\n")
                  f.write("!Vehicle(Car,"+index+")\n")
                  f.write("!Vehicle(Sport,"+index+")\n")
            
            if instance['cr']=="crowded":
                  f.write("Road(Crowded,"+index+")\n")
                  f.write("!Road(Empty,"+index+")\n")
            
            elif instance['cr']=="empty":
                  f.write("!Road(Crowded,"+index+")\n")
                  f.write("Road(Empty,"+index+")\n")
            
            if instance['perception']=="crowded":
                  f.write("Perception(Crowded,"+index+")\n")
                  f.write("!Perception(Empty,"+index+")\n")
            
            elif instance['perception']=="empty":
                  f.write("!Perception(Crowded,"+index+")\n")
                  f.write("Perception(Empty,"+index+")\n")
                    
            if instance['co']=="cooperative":
                  f.write("Cooperative("+index+")\n")
            elif instance['co']=="not cooperative": 
                  f.write("!Cooperative("+index+")\n")
            f.write("\n")
            f.close()
      
      @staticmethod
      def learn_weights(input_file="autocar.mln",output_file="trained.mln",train_data="train_300.db"):
            infer_path = Path.cwd()/'reasoner/learnwts'
            assert Path(infer_path).is_file(), 'learnwts path does not exist'
            #subprocess.check_output([infer_path,"-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query])
            #why this does not work, but in parser pomdp it worked
            subprocess.run(["./learnwts","-g","-i",input_file,"-o",output_file,"-t",train_data],cwd='reasoner')    

      
      @staticmethod
      def infer(mln_file="trained.mln",result_file="autocar_300.result",evidence_file="evidence0810.db",query="Cooperative"):
      
            infer_path = Path.cwd()/'reasoner/infer'
            assert Path(infer_path).is_file(), 'infer path does not exist'
            #subprocess.check_output([infer_path,"-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query])
            #why this does not work, but in parser pomdp it worked
            subprocess.run(["./infer","-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query],cwd='reasoner')    

            
      @staticmethod    
      def read_result(file_name):
            f=open("reasoner/"+file_name,"r")
            cooperative_predict_list=[]
            for line in f:
                  predict=line.split()[-1]
                  cooperative_predict_list.append(predict)
            f.close()
            return cooperative_predict_list

class Planner:
      def __init__(self):
            pomdp_file=Path.cwd()/'model/program.pomdp'
            assert Path(pomdp_file).is_file(), 'POMDP path does not exist'
            
            self.model = Model(pomdp_file=pomdp_file, parsing_print_flag=False)
            #print(self.model.states,self.model.actions)
            
            policy_file = Path.cwd()/'model/program.policy'
            assert Path(policy_file).is_file(), 'POMDP path does not exist'
            #print (policy_file)
            print(self.model.states,self.model.actions)
            
            self.policy = Policy(len(self.model.states),
							     len(self.model.actions),
							     policy_file=policy_file)
      
      def initialize_belief(self,p_co):
            p_co=round(float(p_co),4)
            b1=0.5*p_co
            b2=0.5*(1-p_co)
            b3=0.5*p_co
            b4=0.5*(1-p_co)
            belief=np.array([b1,b2,b3,b4,0.0])
            
            #belief=np.
            #belief=np.array([0.5*p_co,0.5*(1-p_co),0.5*p_co,0.5*(1-p_co),0.0])
            return belief
      #['R_W', 'R_W_not', 'R_not_W', 'R_not_W_not', 'term']
      
      def update_belief(self,a_idx,o_idx,belief):
            belief=np.dot(belief,self.model.trans_mat[a_idx, :])
            #
            belief=[belief[i] * self.model.obs_mat[a_idx, i, o_idx] for i in range(len(self.model.states))]
            belief=belief/sum(belief)
            return belief
      #def update_policy
      
      def observe(self,a_idx,next_state):
            s_idx = self.model.states.index(next_state)
            return np.random.choice(self.model.observations, p= self.model.obs_mat[a_idx,s_idx,:])
      
      def run(self,instance):
            p_co=instance['co_reasoning']
            belief=self.initialize_belief(p_co)
            #gt corresponding to ground truth
            if instance['co']=="cooperative":
                  gt_co=1
            elif instance['co']=='not cooperative':
                  gt_co=0
            gt_belief=self.initialize_belief(gt_co)
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
                  
                  print ('\n\n\nUnderlying state: ', state)
                  instance['pomdp'].append("Underlying state:"+state)
                  
                  print ('action is: ',self.model.actions[a_idx])
                  instance['pomdp'].append("Action:"+self.model.actions[a_idx])
                  
                  next_state = np.random.choice(self.model.states, p=self.model.trans_mat[a_idx,s_idx,:])
                  
                  obs = self.observe(a_idx,next_state)
                  print("obs is:",obs)
                  instance['pomdp'].append("Obs:"+obs)
                  
                  obs_idx = self.model.observations.index(obs)
                  print ('observation is: ',self.model.observations[obs_idx])
                  instance['pomdp'].append("Obs:"+self.model.observations[obs_idx])
                  
                  belief=self.update_belief(a_idx,obs_idx,belief)
                  instance['pomdp'].append(belief.tolist())
                  print(belief)
                  

                  if belief[-1]>0:
                        term=True
                        instance['cost']=r-self.model.reward_mat[a_idx,s_idx] 
                        instance['reward']=r                    
                        if(state=="R_W"):
                              #print("Successful")
                              return "Successful"
                        else:
                              return "Failed"
                        #print("\n")
                  state=next_state
      #def run_pomdp():
      
      #def run_perception_mln():
            

class AbstractSimulator:
      def __init__(self):
            self.weather_list=["sunny","rainy"]
            self.time_period_list=["busy","normal"]
            self.vehicle_type_list=["sport","truck","sedan"]
            self.perception_list=["crowded","empty"]
            self.crowded_list=["crowded","empty"]
            self.willingness_list=["cooperative","not cooperative"]
      
      def sample (self, alist, distribution):
            return np.random.choice(alist, p=distribution)
      
      def minor_sample_cr(self,instance):
            if instance['time_period']=="busy":
                        cr=self.sample(self.crowded_list,[0.9,0.1])          
            elif instance["time_period"]=="normal":
                        cr=self.sample(self.crowded_list,[0.1,0.9]) 
            return cr
      
      def minor_perceive(self,instance,conf1,conf2):
            if instance['cr']=="crowded":
                  perception=self.sample(self.crowded_list,conf1)
            if instance['cr']=="empty":
                  perception=self.sample(self.crowded_list,conf2)
            return perception
      
      def minor_sample_co(self,instance):
            if instance['cr']=="crowded":
                  if instance["weather"]=="rainy":
                        co=self.sample(self.willingness_list,[0.1,0.9])
                  elif instance["weather"]=="sunny":
                        co=self.sample(self.willingness_list,[0.3,0.7])
            
            elif instance['cr']=="empty":
                  if instance["weather"]=="sunny":
                        co=self.sample(self.willingness_list,[0.9,0.1])
                  elif instance["weather"]=="rainy":
                        co=self.sample(self.willingness_list,[0.7,0.3])   
            return co
      
      def minor_create_instance(self):
            instance={}
            instance['weather']=random.choice(self.weather_list)
            instance['time_period']=random.choice(self.time_period_list)
            instance['cr']=self.minor_sample_cr(instance)
            instance['co']=self.minor_sample_co(instance)
            instance['perception']=self.minor_perceive(instance,[1.0,0.0],[0.0,1.0])
            instance['co_reasoning']=None
            return instance
      
      def minor_generate_data(self,num_trials):
            instance_list=[]
            for i in range(0,num_trials):
                  instance=self.minor_create_instance()
                  instance_list.append(instance)
            return instance_list
      
      def minor_check_co(self,instance_list):
            r_b_cr=0
            r_b_cr_co=0
            r_b_cr_nco=0
            for ins in instance_list:
                  if ins['weather']=="rainy" and ins['time_period']=="busy" and ins['perception']=="crowded":
                        r_b_cr+=1
                        if ins["co"]=="cooperative":
                              r_b_cr_co+=1
                        elif ins['co']=="not cooperative":
                              r_b_cr_nco+=1
            assert (r_b_cr_co+r_b_cr_nco)==r_b_cr, 'Probability error'
            return (r_b_cr_co/r_b_cr)


      
      
      def sample_crowded(self,weather,time_period):
            if weather=="sunny":
                  if time_period=="busy":
                        cr=self.sample(self.crowded_list,[0.5,0.5])#0.125 s b cr/em
                  elif time_period=="normal":
                        cr=self.sample(self.crowded_list,[0.1,0.9])#0.025 0.225
            elif weather=="rainy":
                  if time_period=="busy":
                        cr=self.sample(self.crowded_list,[0.9,0.1])
                  elif time_period=="normal":
                        cr=self.sample(self.crowded_list,[0.5,0.5])
            return cr
                  
      def sample_cooperative(self,weather,time_period,vehicle_type,cr):
            if cr=="crowded":
                  if weather=='rainy':
                        if vehicle_type=="truck":
                              co=self.sample(self.willingness_list,[0.1,0.9]) #0.1 for cooperative 0.9 for not cooperative 
                        elif vehicle_type=="sport":
                              co=self.sample(self.willingness_list,[0.2,0.8])
                        elif vehicle_type=="sedan":
                              co=self.sample(self.willingness_list,[0.3,0.7])
                  elif weather=="sunny":
                        co=self.sample(self.willingness_list,[0.5,0.5])
            elif cr=="empty":
                  if weather=='sunny':
                        if vehicle_type=="truck":
                              co=self.sample(self.willingness_list,[0.9,0.1])
                        elif vehicle_type=="sport":
                              co=self.sample(self.willingness_list,[0.8,0.2])
                        elif vehicle_type=="sedan":
                              co=self.sample(self.willingness_list,[0.7,0.3])
                  elif weather=='rainy':
                        co=self.sample(self.willingness_list,[0.5,0.5])
            return co
      


      
      def set_crowded(self,weather,time_period):
            if weather=="sunny":
                  if time_period=="busy":
                        cr=0.5#0.125 s b cr/em
                  elif time_period=="normal":
                        cr=0.1#0.025 0.225
            elif weather=="rainy":
                  if time_period=="busy":
                        cr=0.9
                  elif time_period=="normal":
                        cr=0.5
            return cr
                  
      def set_cooperative(self,weather,time_period,vehicle_type,cr):
            if cr=="crowded":
                  if weather=='rainy':
                        if vehicle_type=="truck":
                              co=0.1 #0.1 for cooperative 0.9 for not cooperative 
                        elif vehicle_type=="sport":
                              co=0.2
                        elif vehicle_type=="sedan":
                              co=0.3
                  elif weather=="sunny":
                        co=0.5
            elif cr=="empty":
                  if weather=='sunny':
                        if vehicle_type=="truck":
                              co=0.9
                        elif vehicle_type=="sport":
                              co=0.8
                        elif vehicle_type=="sedan":
                              co=0.7
                  elif weather=='rainy':
                        co=0.5
            return co      



      
      #def 
      
      def minor_check(self,instance_list):
            r_b=0
            r_n=0
            s_b=0
            s_n=0
            r_b_co=0
            r_n_co=0
            s_b_co=0
            s_n_co=0
            
            for ins in instance_list:
                  if ins['weather']=="rainy":
                        if ins['time_period']=='busy':
                              r_b+=1
                              if ins['co']=="cooperative":
                                    r_b_co+=1
                        elif ins['time_period']=="normal":
                              r_n+=1
                              if ins['co']=="cooperative":
                                    r_n_co+=1
                  
                  if ins['weather']=="sunny":
                        if ins['time_period']=="normal":
                              s_b+=1
                              if ins['co']=="cooperative":
                                    s_n_co+=1
                        
                        if ins['time_period']=="busy":
                              s_n+=1
                              if ins['co']=="cooperative":
                                    s_b_co+=1
            
            p_r_b_co=r_b_co/r_b
            p_r_n_co=r_n_co/r_n
            p_s_n_co=s_n_co/s_n
            p_s_b_co=s_b_co/s_b
            p=[p_r_b_co,p_r_n_co,p_s_n_co,p_s_b_co]
            print(p)
            return p
      
      def minor_plot_sampling(self,start,max,step):
            p_list=[]
            num_list=[]
            instance_list=self.minor_generate_data(1000)
            for i in range(start,max,step):
                  print("The sample size is",i)
                  test_list=self.minor_generate_data(i)
                  p=self.minor_check(test_list)
                  p_list.append(p[0]-0.1)
                  num_list.append(i)
      
            plt.figure()
            plt.plot(num_list,p_list,'.-b')
            plt.xlabel("Number of Samples")
            plt.ylabel("Average probability difference from expected optimal")
            plt.show()
            plt.savefig("performance_of_sampling.pdf")



      def create_instance(self,seed,conf_matrix):
            
            instance={}
            #random.seed(seed)
            instance['weather']=random.choice(self.weather_list)
            #random.seed(seed)
            instance['time_period']=random.choice(self.time_period_list)
            #random.seed(seed)
            instance['vehicle_type']=random.choice(self.vehicle_type_list)
            #random.seed(seed)
            
            instance['cr']=self.sample_crowded(instance['weather'],instance['time_period'])
            instance['cr_soft']=self.set_crowded(instance['weather'],instance['time_period'])
            instance['perception']=self.perceive(instance['cr'],conf_matrix)
            
            instance['co']=self.sample_cooperative(instance['weather'],instance['time_period'],instance["vehicle_type"],instance['cr'])
            instance['co_soft']=self.set_cooperative(instance['weather'],instance['time_period'],instance["vehicle_type"],instance['cr'])
            instance['co_reasoning']=None
            instance['result']=None
            instance['cost']=0
            instance['reward']=0
            instance['belief']=None          
            instance['pomdp']=[]
            #print(instance)
            return instance
      
      def perceive(self,cr,conf_matrix_list):
            p1=[conf_matrix_list[0],conf_matrix_list[1]]
            p2=[conf_matrix_list[2],conf_matrix_list[3]]
            if cr=="crowded":
                  perception=self.sample(self.crowded_list,p1)
            if cr=="empty":
                  perception=self.sample(self.crowded_list,p2)          
            return perception
      
      def create_testdata(self,num_data):
            test_list=[]
            f=open("reasoner/test.db","w")
            for i in range(0,num_data):
                  instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
                  test_list.append(instance)
                  Reasoner.output_evidence(f,instance,str(i))
            f.close()
            
            f=open("test_data.json","w")
            for ins in test_list:
                  js=json.dumps(ins)
                  f.write(js)
                  f.write("\n")
            f.close()
            
            return test_list
      
      def check_cooperative(self,instance_list):
            sum=0
            co=0
            non_co=0
            for ins in instance_list:
                  if ins['weather']=="rainy" and ins['time_period']=="busy" and ins["vehicle_type"]=="truck" and ins["perception"]=="crowded":
                        sum+=1
                        if ins['co']=="cooperative":
                              co+=1
                        elif ins['co']=="not cooperative":
                              non_co+=1
            print(co/sum)
            return co/sum


      def check_conditional_probability(self,instance_list):
            s_b=0
            s_b_cr=0
            s_b_em=0
            s_n=0
            s_n_cr=0
            s_n_em=0
            for ins in instance_list:
                  if ins['weather']=="sunny":
                        if ins['time_period']=="busy":
                              s_b+=1
                              if ins['cr']=="crowded":
                                    s_b_cr+=1
                              else:
                                    s_b_em+=1
                              #cr=self.sample(self.crowded_list,[0.5,0.5])
                        
                        elif ins['time_period']=="normal":
                              s_n+=1
                              if ins['cr']=="crowded":
                                    s_n_cr+=1
                              else:
                                    s_n_em+=1
                              #cr=self.sample(self.crowded_list,[0.1,0.9])
                  
                  elif ins['weather']=="rainy":
                        if ins['time_period']=="busy":
                              cr=self.sample(self.crowded_list,[0.9,0.1])
                        elif ins['time_period']=="normal":
                              cr=self.sample(self.crowded_list,[0.5,0.5])
            
            cr_r_t=0
            cr_r_s=0
            cr_r_c=0
            cr_r_t_co=0
            cr_r_s_co=0
            cr_r_c_co=0

            em_s_t=0
            em_s_s=0
            em_s_c=0
            em_s_t_co=0
            em_s_s_co=0
            em_s_c_co=0

            for ins in instance_list:
                  if ins['cr']=="crowded":
                        if ins['weather']=='rainy':
                              if ins['vehicle_type']=="truck":
                                    cr_r_t+=1
                                    if ins['co']=="cooperative":
                                          cr_r_t_co+=1
                                    #co=self.sample(self.willingness_list,[0.1,0.9]) 
                              elif ins['vehicle_type']=="sport":
                                    cr_r_s+=1
                                    if ins['co']=="cooperative":   
                                          cr_r_s_co+=1
                                    #co=self.sample(self.willingness_list,[0.2,0.8])
                              elif ins['vehicle_type']=="sedan":
                                    cr_r_c+=1
                                    if ins['co']=="cooperative":  
                                          cr_r_c_co+=1
                                    #co=self.sample(self.willingness_list,[0.3,0.7])
                        elif ins['weather']=="sunny":
                              co=self.sample(self.willingness_list,[0.5,0.5])
                  
                  elif ins['cr']=="empty":
                        if ins['weather']=='sunny':
                              if ins['vehicle_type']=="truck":
                                    em_s_t+=1
                                    if ins['co']=="cooperative":
                                          em_s_t_co+=1
                                    #co=self.sample(self.willingness_list,[0.9,0.1])
                              elif ins['vehicle_type']=="sport":
                                    #co=self.sample(self.willingness_list,[0.8,0.2])
                                    em_s_s+=1
                                    if ins['co']=="cooperative":
                                          em_s_s_co+=1
                              elif ins['vehicle_type']=="sedan":
                                    #co=self.sample(self.willingness_list,[0.7,0.3])
                                    em_s_c+=1
                                    if ins['co']=="cooperative":
                                          em_s_c_co+=1
                        elif ins['weather']=='rainy':
                              co=self.sample(self.willingness_list,[0.5,0.5])
            p_cr_r_t_co=cr_r_t_co/cr_r_t
            p_cr_r_s_co=cr_r_s_co/cr_r_s
            p_cr_r_c_co=cr_r_c_co/cr_r_c

            p_em_s_t_co=em_s_t_co/em_s_t
            p_em_s_s_co=em_s_s_co/em_s_s
            p_em_s_c_co=em_s_c_co/em_s_c

            print(p_cr_r_t_co)
            print(p_cr_r_s_co)
            print(p_cr_r_c_co)
            
            print(p_em_s_t_co)
            print(p_em_s_s_co)
            print(p_em_s_c_co)
            return p_cr_r_t_co,p_cr_r_s_co,p_cr_r_c_co,p_em_s_t_co,p_em_s_s_co,p_em_s_c_co
      
      def reason(self,result_file):
            cooperative=Reasoner.read_result(result_file)
            return cooperative
      
      def save_instance(self,instance,result_file="abstract_sim_results"):
            f=open(result_file,"a")
            f.write(instance)
            f.close()
      
      def mln_evaluation(self,num_trial):
            #p_co_list=Reasoner.read_result("0810.result")
            abs_total_reason=0
            abs_total_random=0
            instance_list=[]
            evidence_file_name="0810.db"
            f=open("reasoner/"+evidence_file_name,"w")
            for i in range(0,num_trial):
                  instance=self.create_instance([0.84,0.16,0.84,0.16])
                  instance_list.append(instance)
                  index=str(i)
                  Reasoner.output_evidence(f,instance,index)
                  
            f.close()
            Reasoner.infer(mln_file="trained_300.mln",evidence_file="0810.db",result_file="0810.result")
            p_co_list=Reasoner.read_result("0810.result")
            i=0
            for ins in instance_list:
                  ins["co_reasoning"]=p_co_list[i]
                  p_co=round(float(p_co_list[i]),3)
                  if ins['co']=="cooperative":
                        abs_total_reason+=abs(1-p_co)
                  elif ins['co']=="not cooperative":
                        abs_total_reason+=abs(0-p_co)
                  abs_total_random+=0.5
                  i+=1
            abs_average_reason=abs_total_reason/len(instance_list)
            abs_average_random=abs_total_random/len(instance_list)
            print("Probabilistic soft variation of random is:",abs_average_random)
            print("Probabilistic soft variation of reasoning is:",abs_average_reason)
      
      def run_with_learning(self,batch_size,num_batches):
            batch_count=0
            while batch_count < num_batches:
                  if batch_count==0:
                        self.run_uniform(batch_size)
                  else:
                        self.run(batch_count,batch_size)
                  Reasoner.learn_weights(train_data="train.db")
                  batch_count+=1
      
      def run_uniform(self,num_trial):
            planner=Planner()
            instance_list=[]
            r=0
            cost=0
            average_reward=0
            #average_cost=0
            for i in range(0,num_trial):
                  instance=self.create_instance(i,[0.5,0.5,0.5,0.5])
                  instance['co_reasoning']=0.5
                  instance_list.append(instance)
                  index=str(i)
                  Reasoner.output_data("train.db",instance,index)
                  instance['result']=planner.run(instance)
                  r+=instance['reward']
                  cost+=instance['cost']
            average_reward=r/len(instance_list)
            average_cost=cost/len(instance_list)
            f=open("batch_result.txt",'a')
            f.write("The average cost for btach 0 is:"+str(average_cost)+"\n")
            f.write("The average reward for btach 0 is:"+str(average_reward)+"\n\n")
            f.close()

      
      def run(self,batch_count,batch_size):
            planner=Planner()           
            instance_list=[]
            evidence_file_name="0814.db"
            start_index=batch_count*batch_size
            end_index=start_index+batch_size
            
            f=open("reasoner/"+evidence_file_name,"w")
            for i in range(start_index,end_index):
                  instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
                  instance_list.append(instance)
                  index=str(i)
                  Reasoner.output_data("train.db",instance,index)
                  Reasoner.output_evidence(f,instance,index)
            f.close()
            
            Reasoner.infer(mln_file="trained.mln",evidence_file="0814.db",result_file="0814.result")
            p_cooperative=Reasoner.read_result("0814.result")
            print("The proability of cooperative infered by MLN is",p_cooperative)
            i=0
            for ins in instance_list:
                  ins["co_reasoning"]=p_cooperative[i]
                  i+=1
                  ins['result']=planner.run(ins)
            
            success_count=0
            failure_count=0
            abs_total_reason=0
            r=0
            cost=0
            f=open("abstract_sim_data.json","a")
            for ins in instance_list:
                  if ins["result"]=="Successful":
                        success_count+=1
                  elif ins['result']=="Failed":
                        failure_count+=1  
                  p_co=round(float(ins['co_reasoning']),3)
                  if ins['co']=="cooperative":
                        abs_total_reason+=abs(1-p_co)
                  elif ins['co']=="not cooperative":
                        abs_total_reason+=abs(0-p_co)                   
                  r=r+ins['reward']
                  cost=cost+ins['cost']
                  js=json.dumps(ins)
                  f.write(js)
                  f.write("\n")
            f.close()
            success_rate=success_count/len(instance_list)
            average_reward=r/len(instance_list)
            abs_average_reason=abs_total_reason/len(instance_list)
            average_cost=cost/len(instance_list)
            f=open("batch_result.txt",'a')
            f.write("The success_rate of btach "+str(batch_count)+" is:"+str(success_rate)+"\n")
            f.write("The average reward of btach "+str(batch_count)+" is:"+str(average_reward)+"\n")
            f.write("The average cost of btach "+str(batch_count)+" is:"+str(average_cost)+"\n")
            f.write("Probabilistic soft variation of reasoning is:"+str(abs_average_reason)+"\n\n")
            f.close()
            #print("Success rate is:", success_rate)
            #print("Average reward is:",average_reward)
      

      def evaluate_pomdp(self,num_trial):
            planner=Planner()
            instance_list=[]
            success_count=0
            failure_count=0
            reward_list=[]
            cost_list=[]
            r=0
            cost=0
            
            reward_gt_list=[]
            cost_gt_list=[]
            r_gt=0
            cost_gt=0
            success_gt=0
            failure_gt=0
            
            #we do not the truth and it is like to flip a coin to judge 
            #whether the responding driver is cooperative or not
            f=open("uniform_belief_pomdp.json","w")
            for i in range(0,num_trial):
                  instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
                  instance_list.append(instance)
                  instance['co_reasoning']=0.5
                  instance['result']=planner.run(instance)
                  if instance["result"]=="Successful":
                        success_count+=1
                  elif instance['result']=="Failed":
                        failure_count+=1
                  r=r+instance['reward']
                  cost=cost+instance['cost']
                  reward_list.append(instance['reward'])
                  cost_list.append(instance['cost'])
                  
                  js=json.dumps(instance)
                  f.write(js)
                  f.write("\n")
            f.close()
            
            #we know exactly that the responding driver is cooperative or not
            f=open("ground_truth_belief_pomdp.json","w")
            for ins in instance_list:
                  if ins['co']=="cooperative":
                        ins['co_reasoning']=1
                  elif ins['co']=="not cooperative":
                        ins['co_reasoning']=0
                  
                  ins['result']=planner.run(ins)
                  if ins["result"]=="Successful":
                        success_gt+=1
                  elif ins['result']=="Failed":
                        failure_gt+=1
                  r_gt=r_gt+ins['reward']
                  cost_gt+=ins['cost']
                  reward_gt_list.append(ins['reward'])
                  cost_gt_list.append(ins['cost'])
                  
                  
                  js=json.dumps(ins)
                  f.write(js)
                  f.write("\n")
            f.close()
            
            count=len(instance_list)
            
            success_rate=success_count/count
            average_cost=cost/count
            average_reward=r/count
           
            
            success_rate_gt=success_gt/count
            average_cost_gt=cost_gt/count
            average_reward_gt=r_gt/count
            
            std_reward=st.stdev(reward_list)
            std_reward_gt=st.stdev(reward_gt_list)
            std_cost=st.stdev(cost_list)
            std_cost_gt=st.stdev(cost_gt_list)
            
            
            print(num_trial)
            print("\nSuccess rate with uniform initial belief is:", success_rate)
            print("Failed cases: ",failure_count)
            print("Average cost with uniform initial belief is:%s standard deviation is %s" % (average_cost,std_cost))
            print("Average reward with uniform initial belief is:%s standard deviation is %s" %(average_reward,std_reward))

            print("\nSuccess rate with cooperative with ground truth belief is:", success_rate_gt)
            print("Failed cases: ",failure_gt)
            print("Average cost with with ground truth belief is: %s, standard deviation is %s" %(average_cost_gt,std_cost_gt))
            print("Average reward with with ground truth belief is: %s, standard deviation is %s" %(average_reward_gt,std_reward_gt))
            return average_cost,average_reward,average_cost_gt,average_reward_gt
            #Reasoner.output_evidence("0810.db",instance,index)
      
      def run_mln_pomdp(self):
            planner=Planner()           
            instance_list=[]
            evidence_file_name="0810.db"
            f=open("reasoner/"+evidence_file_name,"w")
            for i in range(0,200):
                  instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
                  instance_list.append(instance)
                  index=str(i)
                  Reasoner.output_evidence(f,instance,index)
            f.close()
            Reasoner.infer(mln_file="trained_300.mln",evidence_file="0810.db",result_file="0810.result")
            p_cooperative=Reasoner.read_result("0810.result")
            print("The proability of cooperative infered by MLN is",p_cooperative)
            i=0
            for ins in instance_list:
                  ins["co_reasoning"]=p_cooperative[i]
                  i+=1
                  ins['result']=planner.run(ins)
            success_count=0
            failure_count=0
            r=0
            f=open("abstract_sim_data.json","w")
            for ins in instance_list:
                  if ins["result"]=="Successful":
                        success_count+=1
                  elif ins['result']=="Failed":
                        failure_count+=1
                  r=r+ins['reward']
                  js=json.dumps(ins)
                  f.write(js)
                  f.write("\n")
            f.close()
            success_rate=success_count/len(instance_list)
            average_reward=r/len(instance_list)
            print("Success rate is:", success_rate)
            print("Average reward is:",average_reward)

      
      def trial(self):
            planner=Planner()
            i=0
            instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
            evidence_file="trial.db"
            f=open("reasoner/"+evidence_file,"w")
            Reasoner.output_evidence(f,instance,'Trial')
            f.close()
            Reasoner.infer(mln_file="trained_300.mln",evidence_file="trial.db",result_file="trial.result")
            p_cooperative=Reasoner.read_result("trial.result")
            instance["co_reasoning"]=p_cooperative[0]
            instance['result']=planner.run(instance)
            print("\n\nThe Ground Truth:",instance["weather"],instance["time_period"],instance["vehicle_type"],instance['cr'],instance['co'])
            print("perception:",instance['perception'],"cooperative_reasoning:",instance['co_reasoning'],"Accumalative reward is:",instance["reward"],"Terminal result is:",)
      
      def evaluate_mln(self,start_index,num_trials,test_list,test_data):
            evidence_file_name="test.db"
            instance_list=[]
            abs_total_reason=0
            abs_total_soft=0
            abs_total_real=0
            probability_gap_list=[]
            end_index=start_index+num_trials
            
            f=open("reasoner/"+evidence_file_name,"w")
            for i in range(start_index,end_index):
                  instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
                  instance_list.append(instance)
                  index=str(i)
                  Reasoner.output_data("train.db",instance,index)
                  Reasoner.output_evidence(f,instance,index)
            f.close()
            
            Reasoner.learn_weights(train_data="train.db")
            Reasoner.infer(mln_file="trained.mln",evidence_file=test_data,result_file="test.result")
            p_co_list=Reasoner.read_result("test.result")
            
            
            f=open("mln_data.json","w")
            i=0
            for ins in test_list:
                  ins["co_reasoning"]=p_co_list[i]
                  p_co=round(float(p_co_list[i]),3)
                  if ins['co']=="cooperative":
                        abs_total_reason+=abs(1-p_co)
                        abs_total_soft+=abs(1-ins['co_soft'])

                  elif ins['co']=="not cooperative":
                        abs_total_reason+=abs(0-p_co)
                        abs_total_soft+=abs(0-ins['co_soft'])
                  
                  abs_total_real+=abs(p_co-ins['co_soft'])
                  i+=1
                  probability_gap_list.append(abs(p_co-ins['co_soft']))               
            f.close()
            
            count=len(instance_list)
            abs_average_reason=abs_total_reason/count
            abs_average_soft=abs_total_soft/count
            abs_average_real=abs_total_real/count
            std_deviation=st.stdev(probability_gap_list)
            #print("Probabilistic soft variation of reasoning is:",abs_average_reason)
            #print("Probabilistic soft variation of soft is:",abs_average_soft)
            print(str(num_trials)+" Average difference between expected probability and mln output is:%s, standard deviation is:%s" %(abs_average_real,std_deviation))
            return abs_average_real
            
            
      def plot_pomdp(self,num_trials):
            cost=[]
            cost_gt=[]
            reward=[]
            reward_gt=[]
            batches=[]
            for i in range(1000,num_trials,500):
                  average_cost,average_reward,average_cost_gt,average_reward_gt=self.evaluate_pomdp(i)
                  cost.append(average_cost)
                  cost_gt.append(average_cost_gt)
                  reward.append(average_reward)
                  reward_gt.append(average_reward_gt)
                  batches.append(i)
      
            plt.figure(figsize=(8,6))
            plt.subplot(121)
            plt.plot(batches,cost,'-b',label="uniform initial belief")
            plt.plot(batches,cost_gt,'-r',label='ground truth initial belief')
            plt.xlabel('Number of trials')
            plt.ylabel('Average Cost')
            plt.legend(loc='upper left', shadow=True, fontsize='xx-small')
            plt.subplot(122)
            plt.plot(batches,reward,'-b',label="uniform initial belief") 
            plt.plot(batches,reward_gt,'-r',label='ground truth initial belief')
            plt.xlabel('Number of trials')
            plt.ylabel('Average Reward')
            plt.legend(loc='upper left', shadow=True, fontsize='xx-small')
            plt.suptitle('Performance of POMDP')
            plt.show()
            plt.savefig("performance_of_pomdp.pdf")
            print("Finished")
      
      def plot_mln(self,max_num):
            probability_gap=[]
            samples_size=[]
            
            #generate test set
            test_list=[]
            f=open("reasoner/test.db","w")
            for i in range(0,3000):
                  instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
                  test_list.append(instance)
                  Reasoner.output_evidence(f,instance,str(i))
            f.close()
            count=len(test_list)
            
            
            for i in range(2000,max_num,1000):
                  average_gap=self.evaluate_mln(i,1000,test_list,"test.db")
                  probability_gap.append(average_gap)
                  samples_size.append(i)
            plt.figure()
            plt.plot(samples_size,probability_gap,'-b')
            plt.xlabel("Number of Samples for learning weights")
            plt.ylabel("Average probability difference from expected optimal")
            plt.suptitle('Performance of MLN')
            plt.show()
            plt.savefig("performance_of_mln.pdf")
      
      #def plot_learning(self,train_size):

      
      def plot_p_r_a(self,train_size):
            test_list=[]
            f=open("reasoner/test.db","w")
            for i in range(0,100):
                  instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
                  test_list.append(instance)
                  Reasoner.output_evidence(f,instance,str(i))
            f.close()
            
            instance_list=[]
            #evidence_file_name='train_pra.db'
            #f=open("reasoner/"+evidence_file_name,"w")
            for i in range(0,train_size):
                  instance=self.create_instance(i,[0.84,0.16,0.84,0.16])
                  instance_list.append(instance)
                  index=str(i)
                  Reasoner.output_data("train_pra.db",instance,index)
                  #.output_evidence(f,instance,index)
            #f.close()
            
            test_data="test.db"
            Reasoner.learn_weights(train_data="train_pra.db",output_file="trained_pra.mln")
            Reasoner.infer(mln_file="trained_pra.mln",evidence_file=test_data,result_file="test_pra.result")
            p_co_list=Reasoner.read_result("test_pra.result")
           
            i=0
            r_pra=0
            cost_pra=0
            planner=Planner()
            for ins in test_list:
                  ins["co_reasoning"]=p_co_list[i]
                  ins['result']=planner.run(ins)
                  i+=1
                  r_pra=r_pra+ins['reward']
                  cost_pra+=ins['cost']
            
            r_a=0
            cost_a=0
            for ins in test_list:
                  ins["co_reasoning"]=0.5
                  ins['result']=planner.run(ins)
                  r_a=r_pra+ins['reward']
                  cost_a+=ins['cost']
            
            r_ra=0
            cost_ra=0
            for i in range(0,train_size):
                  instance=self.create_instance(i,[0.5,0.5,0.5,0.5])
                  index=str(i)
                  Reasoner.output_data("train_ra.db",instance,index)

            Reasoner.learn_weights(train_data="train_ra.db",output_file="trained_ra.mln")
            Reasoner.infer(mln_file="trained_ra.mln",evidence_file=test_data,result_file="test_ra.result")
            p_co_list=Reasoner.read_result("test_ra.result")
            
            i=0
            for ins in test_list:
                  ins["co_reasoning"]=p_co_list[i]
                  ins['result']=planner.run(ins)
                  i+=1
                  r_ra=r_ra+ins['reward']
                  cost_ra+=ins['cost']
            
            reward_list=[]
            cost_list=[]
            
            count=len(test_list)
            avg_reward_pra=r_pra/count
            avg_cost_pra=cost_pra/count
            
            avg_reward_ra=r_ra/count
            avg_cost_ra=cost_ra/count
            
            avg_reward_a=r_a/count
            avg_cost_a=cost_a/count
            
            reward_list.append(avg_reward_a)
            reward_list.append(avg_reward_ra)
            reward_list.append(avg_reward_pra)
            cost_list.append(avg_cost_a)
            cost_list.append(avg_cost_ra)
            cost_list.append(avg_cost_pra)
            names=['A','R+A','P+R+A']
            plt.figure()
            plt.subplot(121)          
            plt.bar(names,reward_list)
            plt.ylabel("Reward")
            
            plt.subplot(122)  
            plt.bar(names,cost_list)
            plt.ylabel("Cost")
            #plt.suptitle('Categorical Plotting')
            plt.show()
            plt.savefig("comparison.pdf")

def main():
      parser = argparse.ArgumentParser(description='Training Settings')
      #parser.add_argument()
      
      sim=AbstractSimulator()
      #sim.minor_plot_sampling(100,8100,100)
      index=0
      results=[]
      samples=[]
      instance={"weather":"rainy","time_period":"busy","perception":"crowded","cr":"crowded","co":None}
      f=open("reasoner/trial.db","w")
      Reasoner.minor_output_evd(f,instance,"1")
      f.close()
      step=200
      data_list=[]
      frequency_list=[]
      for i in range(1,50):
            training_data=sim.minor_generate_data(step)
            data_list=data_list+training_data
            fr=sim.minor_check_co(data_list)
            print("Length is",len(data_list))
            frequency_list.append(fr)
            f=open("reasoner/train.db","a")
            for ins in training_data:
                  Reasoner.minor_output_data(f,ins,str(index))
                  index+=1
            f.close()
            Reasoner.learn_weights(input_file="autocar.mln",output_file="trained.mln",train_data="train.db")
            Reasoner.infer(result_file="test.result",evidence_file="trial.db")
            p_co_list=Reasoner.read_result("test.result")
            results.append((round(float(p_co_list[0]),4)))
            samples.append(i*step)
            print(results,frequency_list,samples)
      plt.figure()
      plt.plot(samples,results,'.-b')
      plt.plot(samples,frequency_list,'.-r')
      plt.xlabel("Number of Samples")
      plt.ylabel("probability")

      plt.savefig("performance_of_MLN.pdf")
      plt.show()


      plt.close()



                  
      


      #for i in range(0,10):

      """
      r_list=[]
      x_list=[]
      zero_list=[]
      test=["a","b"]
      for i in range(100,1000,100):
            count=0
            ratio=0
            
            for y in range(0,i):
                  a=sim.sample(test,[0.9,0.1])
                  if a=="a":
                        count+=1
                  ratio=count/i
                  print(ratio)
           
            r_list.append(ratio)
            x_list.append(i)
            zero_list.append(0.9)
      
      plt.figure()
      plt.plot(x_list,r_list,'-b',x_list,zero_list,"-r")
      plt.xlabel("Number of Samples")
      plt.ylabel("probability")
      plt.show()
      plt.savefig("performance_of_sampling.pdf")
      plt.close()
      """
      
      """
      p1_expected=0.1
      p2_expected=0.2
      p3_expected=0.3
      p4_expected=0.9
      p5_expected=0.8
      p6_expected=0.7
      avg_list=[]
      num_list=[]
      
      for i in range(100,1000,100):
            print("The sample size is",i)
            test_list=sim.create_testdata(i)
            p1,p2,p3,p4,p5,p6=sim.check_conditional_probability(test_list)
            sum=abs(p1-p1_expected)+abs(p2-p2_expected)+abs(p3-p3_expected)+abs(p4-p4_expected)+abs(p5-p5_expected)+abs(p6-p6_expected)
            average=sum/6
            avg_list.append(average)
            num_list.append(i)
      
      plt.figure()
      plt.plot(num_list,avg_list,'-b')
      plt.xlabel("Number of Samples")
      plt.ylabel("Average probability difference from expected optimal")
      plt.show()
      plt.savefig("performance_of_sampling.pdf")
      """

      """
      avg_list=[]
      num_list=[]
      for i in range(500,100000,500):
            test_list=sim.create_testdata(i)
            print("The sample size is",i)
            result=sim.check_cooperative(test_list)
            avg_list.append(abs(result-0.1))
            num_list.append(i)
            plt.figure()
      plt.plot(num_list,avg_list,'-b')
      plt.xlabel("Number of Samples")
      plt.ylabel("probability difference from expected optimal")
      plt.show()
      plt.savefig("performance_of_sampling.pdf")


      #sim.run_with_learning(100,2)
      #sim.run_mln(1200)
      #sim.plot_mln(9001)
      #sim.plot_p_r_a(300)
      #sim.trial()
      #sim.evaluate_pomdp(500)
      #sim.plot_pomdp(9000)
      


            
      
      """
      """
      planner=Planner()
      sim=AbstractSimulator()
      instance_list=[]
      for i in range(0,200):
            instance=sim.create_instance([0.84,0.16,0.84,0.16])
            instance_list.append(instance)
            index=str(i)
            Reasoner.output_evidence("0809.db",instance,index)
      Reasoner.infer(mln_file="trained.mln",evidence_file="0809.db",result_file="0809.result")
      p_cooperative=Reasoner.read_result("0809.result")
      print("The proability of cooperative infered by MLN is",p_cooperative)
      i=0
      for ins in instance_list:      

            ins["co_reasoning"]=p_cooperative[i]
            i+=1
            ins['result']=planner.run(ins)
      success_count=0
      failure_count=0
      f=open("abstract_sim_data.json","w")
      for ins in instance_list:
            if ins["result"]=="Successful":
                  success_count+=1
            elif ins['result']=="Failed":
                  failure_count+=1
            js=json.dumps(ins)
            f.write(js)
            f.write("\n")
      success_rate=success_count/len(instance_list)
      print("Success rate is:", success_rate)
      f.close()
      """

if __name__ == '__main__':
	main()
            
      

            