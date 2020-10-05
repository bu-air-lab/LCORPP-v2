# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:02:17 2020

@author: cckklt
"""
import os
import shutil
import argparse
import json
import random
import math
import numpy as np
from scipy import stats
import statistics as st
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess
from utils.parser import Policy,Solver
from utils.pomdp_parser import Model
from utils.pomdp_solver import generate_policy
np.set_printoptions(precision=4)     # for better belief printing 
import datetime


random.seed(5)
np.random.seed(5)

class Classifier:
      def select_training_images(ins_list,cr_data_path="classifier/lidar_200/train_data/class_B",em_data_path='classifier/lidar_200/train_data/class_A',cr_destination='classifier/lidar_10/train_data/class_B',em_destination='classifier/lidar_10/train_data/class_A'):
            num_cr=0
            num_em=0
            for ins in ins_list:
                  if ins['cr']=="crowded":
                        num_cr+=1
                  elif ins['cr']=="empty":
                        num_em+=1
                  
            cr_image_list=os.listdir(cr_data_path)
            #print(cr_image_list)
            selected_cr_image_list=random.choices(cr_image_list,k=num_cr)
            
            em_image_list=os.listdir(em_data_path)
            #print(em_image_list)
            selected_em_image_list=random.choices(em_image_list,k=num_em)
            
            for f in selected_cr_image_list:
                  shutil.copy(cr_data_path+"/"+f,cr_destination)
            for f in selected_em_image_list:
                  shutil.copy(em_data_path+"/"+f,em_destination)

      def delete_training_images():
            shutil.rmtree("classifier/lidar_10/train_data/class_A")
            shutil.rmtree("classifier/lidar_10/train_data/class_B")
            os.makedirs("classifier/lidar_10/train_data/class_A")
            os.makedirs("classifier/lidar_10/train_data/class_B")



class Reasoner:
      
      @staticmethod
      def minor_output_evd_cr(f,instance,index):
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
            f.write("\n")
      
      @staticmethod
      def minor_output_evd_r_cr(f,instance,index):
            if instance['time_period']=="busy":
                  f.write("Time(Busy,"+index+")\n")
                  f.write("!Time(Normal,"+index+")\n")
            elif instance['time_period']=="normal":
                  f.write("Time(Normal,"+index+")\n")
                  f.write("!Time(Busy,"+index+")\n")           
            
            f.write("!Perception(Crowded,"+index+")\n")
            f.write("Perception(Empty,"+index+")\n")          
            f.write("\n")
      
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
            f.write("\n")
      
      @staticmethod
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
                  #f.write("Cooperative("+index+")\n")
            elif instance['co']=="not cooperative": 
                  f.write("!Cooperative("+index+")\n")
            f.write("\n")
      
      @staticmethod
      #Reasoning only (R): The same as "perception and reasoning", except that the perception module's output is always negative (no vehicle detected).
      def minor_output_data_r(f,instance,index):
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
            
            f.write("!Perception(Crowded,"+index+")\n")
            f.write("Perception(Empty,"+index+")\n")
            
            if instance['co']=="cooperative":
                  f.write("Cooperative("+index+")\n")
            elif instance['co']=="not cooperative": 
                  f.write("!Cooperative("+index+")\n")
            f.write("\n")
      
      @staticmethod
      #Reasoning only (R): The same as "perception and reasoning", except that the perception module's output is always negative (no vehicle detected).
      def minor_output_evd_r(f,instance,index):           
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
            
            f.write("!Perception(Crowded,"+index+")\n")
            f.write("Perception(Empty,"+index+")\n")     
            
            f.write("\n")
            
      
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
      def learn_weights(input_file="autocar.mln",output_file="trained.mln",train_data="train.db"):
            infer_path = Path.cwd()/'reasoner/learnwts'
            assert Path(infer_path).is_file(), 'learnwts path does not exist'
            #subprocess.check_output([infer_path,"-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query])
            #why this does not work, but in parser pomdp it worked
            subprocess.run(["./learnwts","-g","-i",input_file,"-o",output_file,"-t",train_data],cwd='reasoner')    

      
      @staticmethod
      def infer(mln_file="trained.mln",result_file="query.result",evidence_file="query.db",query="Cooperative"):
      
            infer_path = Path.cwd()/'reasoner/infer'
            assert Path(infer_path).is_file(), 'infer path does not exist'
            #subprocess.check_output([infer_path,"-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query])
            #why this does not work, but in parser pomdp it worked
            subprocess.run(["./infer","-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query,"-maxSteps", "200"],cwd='reasoner')

      @staticmethod
      def infer_cr(mln_file="trained.mln",result_file="query_cr.result",evidence_file="query.db",query="Road"):    
            infer_path = Path.cwd()/'reasoner/infer'
            assert Path(infer_path).is_file(), 'infer path does not exist'
            #subprocess.check_output([infer_path,"-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query])
            #why this does not work, but in parser pomdp it worked
            subprocess.run(["./infer","-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query,"-maxSteps", "200"],cwd='reasoner')     

            
      @staticmethod    
      def read_result(file_name):
            f=open("reasoner/"+file_name,"r")
            co_list=[]
            for line in f:
                  predict=line.split()[-1]
                  co=round(float(predict),4)
                  co_list.append(co)
            f.close()
            return co_list
     
      @staticmethod    
      def read_result_cr(file_name):
            f=open("reasoner/"+file_name,"r")
            co_list=[]
            for line in f:
                  if line.find("Empty")!=-1:
                        predict=line.split()[-1]
                        co=round(float(predict),4)
                        co_list.append(co)
            f.close()
            return co_list

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
      
      def initialize_belief(self,p_co,p_cr):
            p_co=round(float(p_co),4)
            p_cr=round(float(p_cr),4)
            b1=0.5*p_co*p_cr
            b2=0.5*p_co*(1-p_cr)
            b3=0.5*(1-p_co)*p_cr
            b4=0.5*(1-p_co)*(1-p_cr)
            b5=0.5*p_co*p_cr
            b6=0.5*p_co*(1-p_cr)
            b7=0.5*(1-p_co)*p_cr
            b8=0.5*(1-p_co)*(1-p_cr)
            belief=np.array([b1,b2,b3,b4,b5,b6,b7,b8,0.0])          
            #belief=np.
            #belief=np.array([0.5*p_co,0.5*(1-p_co),0.5*p_co,0.5*(1-p_co),0.0])
            return belief
      
      def initialize_state_p_distribution(self,p_co,p_cr,p_room):
            p_co=round(float(p_co),4)
            p_cr=round(float(p_cr),4)
            p_room=round(float(p_room),4)
            b1=p_room*p_co*p_cr
            b2=p_room*p_co*(1-p_cr)
            b3=p_room*(1-p_co)*p_cr
            b4=p_room*(1-p_co)*(1-p_cr)
            b5=(1-p_room)*p_co*p_cr
            b6=(1-p_room)*p_co*(1-p_cr)
            b7=(1-p_room)*(1-p_co)*p_cr
            b8=(1-p_room)*(1-p_co)*(1-p_cr)
            belief=np.array([b1,b2,b3,b4,b5,b6,b7,b8,0.0])          
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
            p_cr=instance['cr_reasoning']
            belief=self.initialize_belief(p_co,p_cr)
            #gt corresponding to ground truth
            if instance['co']=="cooperative":
                  gt_co=1.0
            elif instance['co']=='not cooperative':
                  gt_co=0.0
            
            if instance['cr']=="empty":
                  gt_cr=1.0
            elif instance['cr']=='crowded':
                  gt_cr=0.0
            
            if instance['room_left']=="room":
                  gt_room=1.0
            elif instance['room_left']=='no room':
                  gt_room=0.0
            
            gt_belief=self.initialize_state_p_distribution(gt_co,gt_cr,gt_room)
            print("\nGt distribution is",gt_belief)
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
                        if(state=="R_W_C" or state=="R_W_C_not"):
                              #print("Successful")
                              return "Successful"
                        else:
                              return "Failed"
                        #print("\n")
                  state=next_state
            
      def run_pr(self,instance):
            p_co=instance['co_reasoning']
            p_cr=instance['cr_reasoning']
            belief=self.initialize_belief(p_co,p_cr)
            
            #gt corresponding to ground truth
            if instance['co']=="cooperative":
                  gt_co=1
            elif instance['co']=='not cooperative':
                  gt_co=0

            if instance['cr']=="empty":
                  gt_cr=1.0
            elif instance['cr']=='crowded':
                  gt_cr=0.0
            
            if instance['room_left']=="room":
                  gt_room=1.0
            elif instance['room_left']=='no room':
                  gt_room=0.0
            
            gt_belief=self.initialize_state_p_distribution(gt_co,gt_cr,gt_room)
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
                  a_idx=random.choice([0,1])
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
            if(next_state=="R_W_C" or next_state=="R_W_C_not"):
            #print("Successful")
                  return "Successful"
            else:
                  return "Failed"
            

class AbstractSimulator:
      def __init__(self):
            self.weather_list=["sunny","rainy"]
            self.time_period_list=["busy","normal"]
            self.vehicle_type_list=["sport","truck","sedan"]
            self.perception_list=["crowded","empty"]
            self.crowded_list=["crowded","empty"]
            self.willingness_list=["cooperative","not cooperative"]
            self.room_left_list=["room","no room"]
      
      def sample (self, alist, distribution):
            return np.random.choice(alist, p=distribution)
      
      def minor_sample_cr(self,instance):
            if instance['time_period']=="busy":
                        cr=self.sample(self.crowded_list,[0.7,0.3])          
            elif instance["time_period"]=="normal":
                        cr=self.sample(self.crowded_list,[0.3,0.7]) 
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
                        co=self.sample(self.willingness_list,[0.05,0.95])
                  elif instance["weather"]=="sunny":
                        co=self.sample(self.willingness_list,[0.2,0.8])
       
            elif instance['cr']=="empty":
                  if instance["weather"]=="rainy":
                        co=self.sample(self.willingness_list,[0.8,0.2])  
                  elif instance["weather"]=="sunny":
                        co=self.sample(self.willingness_list,[0.95,0.05])
            return co
      
      def minor_create_instance(self,conf1,conf2):
            instance={}
            instance['weather']=random.choice(self.weather_list)
            instance['time_period']=random.choice(self.time_period_list)
            instance['cr']=self.minor_sample_cr(instance)
            instance['co']=self.minor_sample_co(instance)
            instance['perception']=self.minor_perceive(instance,conf1,conf2)
            instance['cr_reasoning']=None
            instance['co_reasoning']=None
            instance['room_left']=random.choice(self.room_left_list)
            return instance
      
      def minor_generate_data(self,num_trials,conf1,conf2):
            instance_list=[]
            for i in range(0,num_trials):
                  instance=self.minor_create_instance(conf1,conf2)
                  instance_list.append(instance)
            return instance_list
      
      def minor_initialize_planning(self,ins):
            ins['pomdp']=None
            ins['belief']=None
            ins['cost']=None
            ins['reward']=None
            ins['result']=None
      
      def minor_check_co(self,instance_list):
            f=[]
            r_b_cr=0
            r_b_cr_co=0
            r_b_cr_nco=0

            r_b_em=0
            r_b_em_co=0
            r_b_em_nco=0

            r_n_cr=0
            r_n_cr_co=0
            r_n_cr_nco=0
            
            r_n_em=0
            r_n_em_co=0
            r_n_em_nco=0

            s_n_cr=0
            s_n_cr_co=0
            s_n_cr_nco=0
            
            s_n_em=0
            s_n_em_co=0
            s_n_em_nco=0

            s_b_cr=0
            s_b_cr_co=0
            s_b_cr_nco=0

            
            s_b_em=0
            s_b_em_co=0
            s_b_em_nco=0

            for ins in instance_list:
                  if ins['weather']=="rainy" and ins['time_period']=="busy" and ins['perception']=="crowded":
                        r_b_cr+=1
                        if ins["co"]=="cooperative":
                              r_b_cr_co+=1
                        elif ins['co']=="not cooperative":
                              r_b_cr_nco+=1
                  
                  if ins['weather']=="rainy" and ins['time_period']=="busy" and ins['perception']=="empty":
                        r_b_em+=1
                        if ins["co"]=="cooperative":
                              r_b_em_co+=1
                        elif ins['co']=="not cooperative":
                              r_b_em_nco+=1
                  
                  elif ins['weather']=="rainy" and ins['time_period']=="normal" and ins['perception']=="crowded":
                        r_n_cr+=1
                        if ins["co"]=="cooperative":
                              r_n_cr_co+=1
                        elif ins['co']=="not cooperative":
                              r_n_cr_nco+=1
                  
                  elif ins['weather']=="rainy" and ins['time_period']=="normal" and ins['perception']=="empty":
                        r_n_em+=1
                        if ins["co"]=="cooperative":
                              r_n_em_co+=1
                        elif ins['co']=="not cooperative":
                              r_n_em_nco+=1
                  
                  elif ins['weather']=="sunny" and ins['time_period']=="normal" and ins['perception']=="corwded":
                        s_n_cr+=1
                        if ins["co"]=="cooperative":
                              s_n_cr_co+=1
                        elif ins['co']=="not cooperative":
                              s_n_cr_nco+=1
                  
                  elif ins['weather']=="sunny" and ins['time_period']=="normal" and ins['perception']=="empty":
                        s_n_em+=1
                        if ins["co"]=="cooperative":
                              s_n_em_co+=1
                        elif ins['co']=="not cooperative":
                              s_n_em_nco+=1
                  
                  elif ins['weather']=="sunny" and ins['time_period']=="busy" and ins['perception']=="crowded":
                        s_b_cr+=1
                        if ins["co"]=="cooperative":
                              s_b_cr_co+=1
                        elif ins['co']=="not cooperative":
                              s_b_cr_nco+=1
                  
                  elif ins['weather']=="sunny" and ins['time_period']=="busy" and ins['perception']=="empty":
                        s_b_em+=1
                        if ins["co"]=="cooperative":
                              s_b_em_co+=1
                        elif ins['co']=="not cooperative":
                              s_b_em_nco+=1
            
            assert (r_b_cr_co+r_b_cr_nco)==r_b_cr, 'Probability error'
            
            p1=r_b_cr_co/(r_b_cr+1)
            p2=r_b_em_co/(r_b_em+1)
            p3=s_b_cr_co/(s_b_cr+1)
            p4=s_b_em_co/(s_b_em+1)
            
            p5=r_n_cr_co/(r_n_cr+1)
            p6=r_n_em_nco/(r_n_em+1)
            p7=s_n_cr_co/(s_n_cr+1)
            p8=s_n_em_nco/(s_n_em+1)

            f=[p1,p2,p3,p4,p5,p6,p7,p8]
            #result=list(np.absolute(np.array(f)-np.array(f_expected)))
            #avg=sum(result)/len(result)
            return f
      
      def minor_check_cr(self,instance_list):
            b_cr=0
            b_cr_cr=0
            b_cr_em=0
            
            b_em=0
            b_em_cr=0
            b_em_em=0

            n_cr=0
            n_cr_cr=0
            n_cr_em=0
            
            n_em=0
            n_em_cr=0
            n_em_em=0
            for ins in instance_list:
                  if ins['time_period']=="busy":
                        if ins['perception']=="crowded":
                              b_cr+=1
                              if ins['cr']=='crowded':
                                    b_cr_cr+=1
                              elif ins['cr']=='empty':
                                    b_cr_em+=1                        
                        elif ins['perception']=="empty":
                              b_em+=1
                              if ins['cr']=='crowded':
                                    b_em_cr+=1
                              elif ins['cr']=='empty':
                                    b_em_em+=1
                  
                  elif ins['time_period']=="normal":                      
                        if ins['perception']=="crowded":
                              n_cr+=1
                              if ins['cr']=='crowded':
                                    n_cr_cr+=1
                              elif ins['cr']=='empty':
                                    n_cr_em+=1
                        
                        elif ins['perception']=="empty":
                              n_em+=1
                              if ins['cr']=='crowded':
                                    n_em_cr+=1
                              elif ins['cr']=='empty':
                                    n_em_em+=1
            p1=round(float(b_cr_em/(b_cr+1)),4)
            p2=round(float(b_em_em/(b_em+1)),4)
            p3=round(float(n_cr_em/(n_cr+1)),4)
            p4=round(float(n_em_em/(n_em+1)),4)
            f=[p1,p2,p3,p4]
            return f

      def minor_check_reasoner(self,instance_list,cr_expect,co_expect,cr_query,co_query):
            cr=self.minor_check_cr(instance_list)
            co=self.minor_check_co(instance_list)
            
            f=open("prob.txt","a")
            f.write("\nCr stat"+str(cr))
            
            f.write("\nCr Reasoner"+str(cr_query))
            f.write("\nCr Expected"+str(cr_expect))
            
            f.write("\nCr stat"+str(co))
            f.write("\nCr Reasoner"+str(co_query))
            f.write("\nCr Expected"+str(co_expect))
            f.close()
      
      def minor_check_prob(self,list1,list2):
            result=list(np.absolute(np.array(list1)-np.array(list2)))
            avg=sum(result)/len(result)
            return avg



      
      def minor_generate_query_evidence(self):
            instance1={"time_period":"busy","weather":"rainy","perception":"crowded","cr":"crowded","co":None}
            instance2={"time_period":"busy","weather":"rainy","perception":"empty","cr":"crowded","co":None}
            instance3={"time_period":"busy","weather":"sunny","perception":"crowded","cr":"empty","co":None}
            instance4={"time_period":"busy","weather":"sunny","perception":"empty","cr":"empty","co":None}
            
            instance5={"time_period":"normal","weather":"rainy","perception":"crowded","cr":"crowded","co":None}
            instance6={"time_period":"normal","weather":"rainy","perception":"empty","cr":"crowded","co":None}
            instance7={"time_period":"normal","weather":"sunny","perception":"crowded","cr":"crowded","co":None}
            instance8={"time_period":"normal","weather":"sunny","perception":"empty","cr":"crowded","co":None}
            
            
            f=open("reasoner/query.db","w")
            Reasoner.minor_output_evd(f,instance1,"1")
            Reasoner.minor_output_evd(f,instance2,"2")
            Reasoner.minor_output_evd(f,instance3,"3")
            Reasoner.minor_output_evd(f,instance4,"4")
            Reasoner.minor_output_evd(f,instance5,"5")
            Reasoner.minor_output_evd(f,instance6,"6")
            Reasoner.minor_output_evd(f,instance7,"7")
            Reasoner.minor_output_evd(f,instance8,"8")
            f.close()

            f=open("reasoner/query_r.db","w")
            Reasoner.minor_output_evd_r(f,instance1,"1")

            Reasoner.minor_output_evd_r(f,instance3,"2")

            Reasoner.minor_output_evd_r(f,instance5,"3")

            Reasoner.minor_output_evd_r(f,instance7,"4")

            f.close()

            f=open("reasoner/query_cr.db","w")
            Reasoner.minor_output_evd_cr(f,instance1,"1")

            Reasoner.minor_output_evd_cr(f,instance2,"2")

            Reasoner.minor_output_evd_cr(f,instance5,"3")

            Reasoner.minor_output_evd_cr(f,instance6,"4")
            f.close()

            f=open("reasoner/query_r_cr.db","w")
            Reasoner.minor_output_evd_r_cr(f,instance1,"1")

            Reasoner.minor_output_evd_r_cr(f,instance2,"5")
            f.close()
      
      def minor_query_reasoner(self,q,instance_list):
            for ins in instance_list:
                  if ins['time_period']=="busy":
                        if ins['weather']=="rainy":
                              if ins['perception']=="crowded":
                                    ins["co_reasoning"]=q[0]
                              elif ins['perception']=="empty":
                                    ins["co_reasoning"]=q[1]
                  
                        elif ins['weather']=="sunny" :
                              if ins['perception']=="crowded":
                                    ins["co_reasoning"]=q[2]
                              elif ins['perception']=="empty":
                                    ins["co_reasoning"]=q[3]

                  elif ins['time_period']=="normal":
                        if ins['weather']=="rainy":
                              if ins['perception']=="crowded":
                                    ins["co_reasoning"]=q[4]
                              elif ins['perception']=="empty":
                                    ins["co_reasoning"]=q[5]
                  
                        elif ins['weather']=="sunny":
                              if ins['perception']=="crowded":
                                    ins["co_reasoning"]=q[6]
                              elif ins['perception']=="empty":
                                    ins["co_reasoning"]=q[7]
      
      def minor_query_reasoner_r(self,q,instance_list):
            for ins in instance_list:
                  if ins['time_period']=="busy":
                        if ins['weather']=="rainy":
                              ins["co_reasoning"]=q[0]
                        elif ins['weather']=="sunny" :
                              ins["co_reasoning"]=q[1]
                  
                  elif ins['time_period']=="normal":
                        if ins['weather']=="rainy":
                              ins["co_reasoning"]=q[2]                 
                        elif ins['weather']=="sunny":
                              ins["co_reasoning"]=q[3]
      
      def minor_query_congestion(self,q,instance_list):          
            for ins in instance_list:
                  if ins['time_period']=="busy":
                        if ins['perception']=='crowded':
                              ins["cr_reasoning"]=q[0]
                        elif ins['perception']=='empty':
                              ins["cr_reasoning"]=q[1]             
                  elif ins['time_period']=="normal":
                        if ins['perception']=='crowded':
                              ins["cr_reasoning"]=q[2]
                        elif ins['perception']=='empty':
                              ins["cr_reasoning"]=q[3]
      
      def minor_query_congestion_r(self,q,instance_list):          
            for ins in instance_list:
                  if ins['time_period']=="busy":
                              ins["cr_reasoning"]=q[0]                             
                  elif ins['time_period']=="normal":
                              ins["cr_reasoning"]=q[1]
      
      def minor_get_metrics(self,ins_list):
            reward=0
            cost=0
            success=0
            for ins in ins_list:
                  reward+=ins['reward']
                  cost+=ins['cost']
                  if ins['result']=="Successful":
                        success+=1
            l=len(ins_list)
            avg_r=reward/l
            avg_c=cost/l
            avg_s=success/l
            return avg_r,avg_c,avg_s
      
      def minor_create_testdata(self,num_data,conf1,conf2):
            test_list=[]
            #f=open("reasoner/test.db","w")
            for i in range(0,num_data):
                  instance=self.minor_create_instance(conf1,conf2)
                  test_list.append(instance)
                  #Reasoner.minor_output_evd(f,instance,str(i))
            #f.close()     
            return test_list

      def save_data(self,instances,name):
            f=open(name+".json","w")
            for ins in instances:
                  js=json.dumps(ins)
                  f.write(js)
                  f.write("\n")
            f.close()  
      
      def minor_planning(self,planner,instance_list):
            for ins in instance_list:
                  self.minor_initialize_planning(ins)
                  ins['result']=planner.run(ins)
      
      def minor_planning_pr(self,planner,instance_list):
            for ins in instance_list:
                  self.minor_initialize_planning(ins)
                  ins['result']=planner.run_pr(ins) 
      
      def minor_plot_trend(self,xlist,ylist,xlabel,ylabel,name):
            plt.figure()
            plt.plot(xlist,ylist,'.-b')
            #plt.plot(samples,frequency_list,'.-r')
            #plt.plot(samples,avg_list,'.-g')
            #plt.plot(samples,reasoning_list,'.-y')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.savefig(name)
            plt.show()


            plt.close()
      
      #  Planning only (A): purely POMDP-based, assuming that people are all cooperative at the beginning of the interaction.
      def minor_a(self,planner,instance_list):
            i=0
            for ins in instance_list:
                  i+=1
                  print("\n\nWith uniform distribution reasoning")
                  self.minor_initialize_planning(ins)
                  ins["co_reasoning"]=0.5
                  ins["cr_reasoning"]=0.5
                  print("Instance:",i,ins)
                  ins["result"]=planner.run(ins)
                  print("Instance:",i,ins['result'],ins['cost'],ins['reward'])
                  
      
      def minor_a_gt(self,planner,instance_list):
            i=0
            for ins in instance_list:
                  i+=1
                  print("\n\nWith accurate logical reasoning")
                  self.minor_initialize_planning(ins)
                  if ins['co']=="cooperative":
                        ins["co_reasoning"]=0.95
                  elif ins['co']=='not cooperative':
                        ins["co_reasoning"]=0.05                
                  if ins['cr']=="empty":
                        ins["cr_reasoning"]=0.95
                  elif ins['cr']=='crowded':
                        ins["cr_reasoning"]=0.05               
                  print("Instance:",i,ins)
                  ins["result"]=planner.run(ins)
                  print("Instance:",i,ins['result'],ins['cost'],ins['reward'])
      
      def minor_a_test_err(self,instance_list):
            planner=Planner()
            a_r=[]
            a_c=[]
            a_s=[]
            a_r_gt=[]
            a_c_gt=[]
            a_s_gt=[]
            for i in range(5):
                  self.minor_a(planner,instance_list)
                  self.save_data(instance_list,"A")
                  avg_r_a,avg_c_a,avg_s_a=self.minor_get_metrics(instance_list)
                  a_r.append(avg_r_a)
                  a_c.append(avg_c_a)
                  a_s.append(avg_s_a)

                  self.minor_a_gt(planner,instance_list)
                  self.save_data(instance_list,"A_gt")
                  avg_r_a_gt,avg_c_a_gt,avg_s_a_gt=self.minor_get_metrics(instance_list)
                  a_r_gt.append(avg_r_a_gt)
                  a_c_gt.append(avg_c_a_gt)
                  a_s_gt.append(avg_s_a_gt)

                  a_r_d=avg_r_a_gt-avg_r_a
                  a_c_d=avg_c_a-avg_c_a_gt
                  a_s_d=avg_s_a_gt-avg_s_a

                  print("The reward gap is:",a_r_d)
                  print("The cost gap is:",a_c_d)
                  print("The success gap is:",a_s_d)

            mean_a_r=sum(a_r)/len(a_r)
            mean_a_c=sum(a_c)/len(a_c)
            mean_a_s=sum(a_s)/len(a_s)

            mean_a_r_gt=sum(a_r_gt)/len(a_r_gt)
            mean_a_c_gt=sum(a_c_gt)/len(a_c_gt)
            mean_a_s_gt=sum(a_s_gt)/len(a_s_gt)

            r=[mean_a_r,mean_a_r_gt]
            c=[mean_a_c,mean_a_c_gt]
            s=[mean_a_s,mean_a_s_gt]
            err_r=[st.stdev(a_r),st.stdev(a_r_gt)]
            err_c=[st.stdev(a_c),st.stdev(a_c_gt)]
            err_s=[st.stdev(a_s),st.stdev(a_s_gt)]


            baselines=["A","A_gt"]
            plt.figure()
            #plt.subplots()
            plt.bar(baselines,r,yerr=err_r)
            plt.ylabel("Reward")
            plt.ylim(50,)
            plt.savefig("pomdp_r.pdf")
            plt.close()
            
            plt.figure()
            #plt.subplots()
            plt.bar(baselines,c,yerr=err_c)
            plt.ylabel("Cost")
            plt.ylim(5,)
            plt.savefig("pomdp_c.pdf")

            plt.figure()
            #plt.subplots()
            plt.bar(baselines,s,yerr=err_s)
            plt.ylabel("Successrate")
            plt.ylim(0.8,1)
            plt.savefig("pomdp_s.pdf")
            plt.close()
      
      def minor_a_test(self,instance_list):
            planner=Planner()
            a_r=[]
            a_c=[]
            a_s=[]
            self.minor_a(planner,instance_list)
            self.save_data(instance_list,"A")
            avg_r_a,avg_c_a,avg_s_a=self.minor_get_metrics(instance_list)
            a_r.append(avg_r_a)
            a_c.append(avg_c_a)
            a_s.append(avg_s_a)

            self.minor_a_gt(planner,instance_list)
            self.save_data(instance_list,"A_gt")
            avg_r_a_gt,avg_c_a_gt,avg_s_a_gt=self.minor_get_metrics(instance_list)
            a_r.append(avg_r_a_gt)
            a_c.append(avg_c_a_gt)
            a_s.append(avg_s_a_gt)

            a_r_d=avg_r_a_gt-avg_r_a
            a_c_d=avg_c_a-avg_c_a_gt
            a_s_d=avg_s_a_gt-avg_s_a

            print("The reward gap is:",a_r_d)
            print("The cost gap is:",a_c_d)
            print("The success gap is:",a_s_d)

            baselines=["A","A_gt"]
            plt.figure()
            #plt.subplots()
            plt.bar(baselines,a_r)
            plt.ylabel("Reward")
            #plt.ylim(50,)
            plt.savefig("pomdp_r.pdf")
            plt.close()
            
            plt.figure()
            #plt.subplots()
            plt.bar(baselines,a_c)
            plt.ylabel("Cost")
            #plt.ylim(5,)
            plt.savefig("pomdp_c.pdf")

            plt.figure()
            #plt.subplots()
            plt.bar(baselines,a_s)
            plt.ylabel("Successrate")
            #plt.ylim(0.8,1)
            plt.savefig("pomdp_s.pdf")
            plt.close()


      
      def minor_del_reasoner_file(self):
            if os.path.exists("reasoner/test.db"):
                  os.remove("reasoner/test.db")
            if os.path.exists("reasoner/train.db"):
                  os.remove("reasoner/train.db")
            if os.path.exists("reasoner/train_r.db"):
                  os.remove("reasoner/train_r.db")
            if os.path.exists("reasoner/trained.mln"):
                  os.remove("reasoner/trained.mln")
            if os.path.exists("reasoner/query.db"):
                  os.remove("reasoner/query.db")
            if os.path.exists("reasoner/query_r.db"):
                  os.remove("reasoner/query_r.db")
            if os.path.exists("reasoner/query.result"):
                  os.remove("reasoner/query.result")
            if os.path.exists("reasoner/query_r.result"):
                  os.remove("reasoner/query_r.result")  
            
            if os.path.exists("reasoner/query_cr.db"):
                  os.remove("reasoner/query_cr.db")  
            
            if os.path.exists("reasoner/query_cr.result"):
                  os.remove("reasoner/query_cr.result")  

            if os.path.exists("reasoner/query_r_cr.db"):
                  os.remove("reasoner/query_r_cr.db")  
            
            if os.path.exists("reasoner/query_r_cr.result"):
                  os.remove("reasoner/query_r_cr.result") 


      #def minor_run(self,test_data,step,batch_num,steps):
      def minor_run(self,test_data,steps,conf):
            self.minor_del_reasoner_file()
            self.minor_generate_query_evidence()
            conf1=[0.9,0.1]
            conf2=[0.1,0.9]
            cr_expect=[]
            co_expect=[]
            
            #test_data=self.minor_create_testdata(3000,conf1,conf2)
            test_data=test_data
            cr_test=self.minor_check_cr(test_data)
            co_test=self.minor_check_co(test_data)
            
            data_list=[]
            index=0
            index_r=0
            planner=Planner()
            lpra_r=[]
            lpra_c=[]
            lpra_s=[]
            a_r=[]
            a_c=[]
            a_s=[]
            
            p_learn_r=[]
            p_learn_s=[]

            r_learn_r=[]
            r_learn_s=[]

            samples=[]
            cr_diffs=[]
            co_diffs=[]
            conf_index=0
            #for i in range(1,batch_num):
            for step in steps:
                  #samples.append(step+10)

                  #generate training data
                  #training_data=self.minor_generate_data(step,conf1,conf2)
                  training_data=random.choices(test_data,k=step)
                  data_list=data_list+training_data
                  print("Length is",len(data_list))
                  samples.append(math.log(len(data_list)))
                  
                  
                  #generate training evidences for LPRA
                  f=open("reasoner/train.db","w")
                  for ins in data_list:
                        #perception
                        ins['perception']=self.minor_perceive(ins,conf[conf_index][0],conf[conf_index][1])
                        Reasoner.minor_output_data(f,ins,str(index))
                        index+=1
                  f.close()
                  conf_index+=1
                  #generate training evidence for RA
                  f=open("reasoner/train_r.db","a")
                  for ins in training_data:
                        Reasoner.minor_output_data_r(f,ins,str(index_r))
                        index_r+=1
                  f.close()
                  
                  #learn the weights from training data for LPRA
                  Reasoner.learn_weights(input_file="autocar.mln",output_file="trained.mln",train_data="train.db")
                  Reasoner.infer_cr(result_file="query_cr.result",evidence_file="query_cr.db")
                  Reasoner.infer(result_file="query.result",evidence_file="query.db")

                  #learn the weights from training data for RA
                  Reasoner.learn_weights(input_file="autocar_r.mln",output_file="trained_r.mln",train_data="train_r.db")
                  Reasoner.infer_cr(result_file="query_r_cr.result",evidence_file="query_r_cr.db",mln_file="trained_r.mln")
                  Reasoner.infer(result_file="query_r.result",evidence_file="query_r.db",mln_file="trained_r.mln")
                  
                  #get the query results for LPRA
                  co_query=Reasoner.read_result("query.result")
                  cr_query=Reasoner.read_result_cr("query_cr.result")
                  
                  #self.minor_check_reasoner(training_data,cr_expect,co_expect,cr_query,co_query)
                  #check the reasoner learning result
                  cr_diff=self.minor_check_prob(cr_query,cr_test)
                  co_diff=self.minor_check_prob(co_query,co_test)
                  co_diffs.append(co_diff)
                  cr_diffs.append(cr_diff)
                  
                  #get the query resultf for RA
                  co_query_r=Reasoner.read_result("query_r.result")
                  cr_query_r=Reasoner.read_result_cr("query_r_cr.result")
                  
                  print(co_query)
                  print(co_query_r)
                  
                  # map the LPRA reasoning results to test data
                  self.minor_query_reasoner(co_query,test_data)
                  self.minor_query_congestion(cr_query,test_data)
                  self.minor_planning(planner,test_data)
                  self.save_data(test_data,"lpra")

                  avg_r,avg_c,avg_s=self.minor_get_metrics(test_data)
                  
                  lpra_r.append(avg_r)
                  lpra_c.append(avg_c)
                  lpra_s.append(avg_s)

                  # map the PR results to test data
                  self.minor_planning_pr(planner,test_data)
                  self.save_data(test_data,"pr")
                  avg_r_pr,avg_c_pr,avg_s_pr=self.minor_get_metrics(test_data)


                  # map the RA reasoning results to test data
                  self.minor_query_reasoner_r(co_query,test_data)
                  self.minor_query_congestion_r(cr_query,test_data)
                  self.minor_planning(planner,test_data)
                  self.save_data(test_data,"RA")
                  avg_r_ra,avg_c_ra,avg_s_ra=self.minor_get_metrics(test_data)                     

            print(cr_query)
            a_r.append(avg_r)
            a_c.append(avg_c)
            a_s.append(avg_s)

            a_r.append(avg_r_pr)
            a_c.append(avg_c_pr)
            a_s.append(avg_s_pr)
            
            a_r.append(avg_r_ra)
            a_c.append(avg_c_ra)
            a_s.append(avg_s_ra)

            #plot the learning trend for LPRA   
            print("Samples is",samples)    
            self.minor_plot_trend(samples,lpra_r,"Training Data Size","Average Reward on Test Data",str(datetime.datetime.now())+"reward.pdf")
            self.minor_plot_trend(samples,lpra_c,"Training Data Size","Average Cost on Test Data",str(datetime.datetime.now())+"cost.pdf")
            self.minor_plot_trend(samples,lpra_s,"Training Data Size","Success rate on Test Data",str(datetime.datetime.now())+"success_rate.pdf")
            self.minor_plot_trend(samples,co_diffs,"Training Data Size","avergae prediction differences","co_trend.pdf")
            self.minor_plot_trend(samples,cr_diffs,"Training Data Size","avergae prediction differences","cr_trend.pdf")

            #apply only A on test data
            self.minor_a(planner,test_data)
            self.save_data(test_data,"A")
            avg_r_a,avg_c_a,avg_s_a=self.minor_get_metrics(test_data)

            a_r.append(avg_r_a)
            a_c.append(avg_c_a)
            a_s.append(avg_s_a)

            print("average is",a_r,a_c,a_s)

            baselines=["LPRA","PRA-","RA","A"]
            print(cr_query)
            
            plt.figure()
            #plt.subplots()
            plt.bar(baselines,a_r)
            plt.ylabel("Reward")
            #plt.ylim()
            plt.savefig("a_r.pdf")
            plt.close()
            
            plt.figure()
            #plt.subplots()
            plt.bar(baselines,a_c)
            plt.ylabel("Cost")
            #plt.ylim()
            plt.savefig("a_c.pdf")

            plt.figure()
            #plt.subplots()
            plt.bar(baselines,a_s)
            plt.ylabel("Successrate")
            #plt.ylim()
            plt.savefig("a_s.pdf")
            plt.close()
            

            return a_r,a_c,a_s,lpra_r,lpra_c,lpra_s,samples
      
      def minor_multi_run(self,test_data,steps,conf):
      #def minor_multi_run(self,test_data,step,batch_num):
            LPRA_r=[]
            LPRA_c=[]
            LPRA_s=[]
            PR_r=[]
            PR_c=[]
            PR_s=[]
            RA_r=[]
            RA_c=[]
            RA_s=[]
            A_r=[]
            A_c=[]
            A_s=[]
            trends_r=[]
            trends_c=[]
            trends_s=[]
            #sim.minor_plot_sampling(100,8100,100)
            for i in range(9):
                  self.minor_del_reasoner_file()       
                  a_r,a_c,a_s,l_trend_r,l_trend_c,l_trend_s,datasizes=self.minor_run(test_data,steps,conf)
                  
                  trends_r.append(l_trend_r)
                  trends_c.append(l_trend_c)
                  trends_s.append(l_trend_s)

                  LPRA_r.append(a_r[0])
                  LPRA_c.append(a_c[0])
                  LPRA_s.append(a_s[0])
                  PR_r.append(a_r[1])
                  PR_c.append(a_c[1])
                  PR_s.append(a_s[1])
                  RA_r.append(a_r[2])
                  RA_c.append(a_c[2])
                  RA_s.append(a_s[2])
                  A_r.append(a_r[3])
                  A_c.append(a_c[3])
                  A_s.append(a_s[3])
            
            print("trend r is",trends_r)
            print("trend s is",trends_s)
            
            batch_num=len(steps)+1
            err_r=[[trends_r[x][y] for x in range(len(trends_r))] for y in range(batch_num-1)]
            err_c=[[trends_c[x][y] for x in range(len(trends_c))] for y in range(batch_num-1)]
            err_s=[[trends_s[x][y] for x in range(len(trends_s))] for y in range(batch_num-1)]
            mean_trend_r=[sum(err_r[x])/len(err_r[x]) for x in range(batch_num-1)]
            mean_trend_c=[sum(err_c[x])/len(err_c[x]) for x in range(batch_num-1)]
            mean_trend_s=[sum(err_s[x])/len(err_s[x]) for x in range(batch_num-1)]
            std_trend_r=[st.stdev(err_r[x]) for x in range(batch_num-1)]
            std_trend_c=[st.stdev(err_c[x]) for x in range(batch_num-1)]
            std_trend_s=[st.stdev(err_s[x]) for x in range(batch_num-1)]
            
            #datasizes=[10,20,40,80,160]
            plt.figure()
            #plt.subplots()
            plt.errorbar(datasizes,mean_trend_r,yerr=std_trend_r,capsize=2)
            plt.ylabel("Reward")
            #plt.ylim(30,)
            plt.savefig("LPRA_reward.pdf")
            plt.close()

            plt.figure()
            #plt.subplots()
            plt.errorbar(datasizes,mean_trend_c,yerr=std_trend_c,capsize=2)
            plt.ylabel("Cost")
            #plt.ylim(50,80)
            plt.savefig("LPRA_cost.pdf")
            plt.close()

            
            plt.figure()
            #plt.subplots()
            plt.errorbar(datasizes,mean_trend_s,yerr=std_trend_s,capsize=2)
            plt.ylabel("Success rate")
            #plt.ylim(0,6,)
            plt.savefig("LPRA_successrate.pdf")
            plt.close()

            
            mean_LPRA_r=sum(LPRA_r)/len(LPRA_r)
            mean_LPRA_c=sum(LPRA_c)/len(LPRA_c)
            mean_LPRA_s=sum(LPRA_s)/len(LPRA_s)
            mean_PR_r=sum(PR_r)/len(PR_r)
            mean_PR_c=sum(PR_c)/len(PR_c)
            mean_PR_s=sum(PR_s)/len(PR_s)
            mean_RA_r=sum(RA_r)/len(RA_r)
            mean_RA_c=sum(RA_c)/len(RA_c)
            mean_RA_s=sum(RA_s)/len(RA_s)
            mean_A_r=sum(A_r)/len(A_r)
            mean_A_c=sum(A_c)/len(A_c)
            mean_A_s=sum(A_s)/len(A_s)

            std_LPRA_r=st.stdev(LPRA_r)
            std_LPRA_c=st.stdev(LPRA_c)
            std_LPRA_s=st.stdev(LPRA_s)
            std_PR_r=st.stdev(PR_r)
            std_PR_c=st.stdev(PR_c)
            std_PR_s=st.stdev(PR_s)
            std_RA_r=st.stdev(RA_r)
            std_RA_c=st.stdev(RA_c)
            std_RA_s=st.stdev(RA_s)
            std_A_r=st.stdev(A_r)
            std_A_c=st.stdev(A_c)
            std_A_s=st.stdev(A_s)
      
            error_r=[std_LPRA_r,std_PR_r,std_RA_r,std_A_r]
            error_c=[std_LPRA_c,std_PR_c,std_RA_c,std_A_c]
            error_s=[std_LPRA_s,std_PR_s,std_RA_s,std_A_s]
      
            reward=[mean_LPRA_r,mean_PR_r,mean_RA_r,mean_A_r]
            cost=[mean_LPRA_c,mean_PR_c,mean_RA_c,mean_A_c]
            success=[mean_LPRA_s,mean_PR_s,mean_RA_s,mean_A_s]
            baselines=["LPRA","LPRA-","LRA","A"]
            
            print("LPRA:",LPRA_r,LPRA_c,LPRA_s)
            print("PR is:",PR_r,PR_c,PR_s)
            print("RA is:",RA_r,RA_c,RA_s)
            print("A is:",A_r,A_c,A_s)
            print(reward,cost,success)
            print(stats.ttest_ind(LPRA_r,RA_r))
            plt.figure()
            #plt.subplots()
            plt.bar(baselines,reward,yerr=error_r,capsize=2,width=0.6)
            plt.ylabel("Reward")
            plt.ylim(35,)
            plt.savefig("reward_comparison.pdf")
            plt.close()

            plt.figure()
            #plt.subplots()
            plt.bar(baselines,cost,yerr=error_c,capsize=2,width=0.6)
            plt.ylabel("Cost")
            plt.ylim(5,)
            plt.savefig("cost_comparison.pdf")

            plt.figure()
            #plt.subplots()
            plt.bar(baselines,success,yerr=error_s,capsize=2,width=0.6)
            plt.ylabel("Successrate")
            plt.ylim(0.8,)
            plt.savefig("success_comparison.pdf")
            plt.close()


def main():
      parser = argparse.ArgumentParser(description='Training Settings')
      #parser.add_argument()

      conf1=[0.9,0.1]
      conf2=[0.1,0.9]
      sim=AbstractSimulator()
      
      steps=[10,10,30,50,100,200,200]
      conf=[[[0.5,0.5],[0.5,0.5]],[[0.59,0.41],[0.41,0.59]],[[0.67,0.33],[0.33,0.67]],[[0.84,0.16],[0.16,0.84]],[[0.86,0.14],[0.14,0.86]],[[0.86,0.14],[0.14,0.86]],[[0.86,0.14],[0.14,0.86]]]
      test_data=sim.minor_create_testdata(1200,conf1,conf2)
      sim.minor_a_test(test_data)
      #list_ins=sim.minor_create_testdata(10,conf1,conf2)
      #sim.minor_multi_run(test_data,steps,conf)
      #Classifier.select_training_images(list_ins)
      #Classifier.delete_training_images()
      #sim.minor_run(test_data,steps,conf)
      
      """
      batch_num=5
      trends_r=[[1,2,3,4,5],[2,3,4,5,6],[1,2,3,4,5],[2,3,4,5,6],[2,3,4,5,6]]
      trends_c=[[1,2,3,4,5],[2,3,4,5,6],[1,2,3,4,5],[2,3,4,5,6],[2,3,4,5,6]]
      trends_s=[[1,2,3,4,5],[2,3,4,5,6],[1,2,3,4,5],[2,3,4,5,6],[2,3,4,5,6]]

      err_r=[[trends_r[x][y] for x in range(5)] for y in range(batch_num)]
      err_c=[[trends_c[x][y] for x in range(5)] for y in range(batch_num)]
      err_s=[[trends_s[x][y] for x in range(5)] for y in range(batch_num)]
      print(err_r)
      mean_trend_r=[sum(err_r[x])/len(err_r[x]) for x in range(batch_num)]
      mean_trend_c=[sum(err_c[x])/len(err_c[x]) for x in range(batch_num)]
      mean_trend_s=[sum(err_s[x])/len(err_s[x]) for x in range(batch_num)]
      std_trend_r=[st.stdev(err_r[x]) for x in range(batch_num)]
      std_trend_c=[st.stdev(err_c[x]) for x in range(batch_num)]
      std_trend_s=[st.stdev(err_s[x]) for x in range(batch_num)]
            
      datasizes=[i*200 for i in range(1,batch_num+1)]
      plt.figure()
      #plt.subplots()
      plt.errorbar(datasizes,mean_trend_r,yerr=std_trend_r,capsize=2)
      plt.ylabel("Reward")
      #plt.ylim(50,80)
      plt.savefig("trend_r.pdf")
      plt.close()
      """
      

      
      """
      conf1=[0.9,0.1]
      conf2=[0.1,0.9]
      test_data=sim.minor_create_testdata(3000,conf1,conf2)
      planner=Planner()
      sim.minor_a_test(planner,test_data)
      """
 
      
      #sim.minor_run(100,2)
      

      """
      LPRA_r=[]
      LPRA_c=[]
      LPRA_s=[]
      PR_r=[]
      PR_c=[]
      PR_s=[]
      RA_r=[]
      RA_c=[]
      RA_s=[]
      A_r=[]
      A_c=[]
      A_s=[]
      #sim.minor_plot_sampling(100,8100,100)
      for i in range(5):
            if os.path.exists("reasoner/test.db"):
                  os.remove("reasoner/test.db")
            if os.path.exists("reasoner/train.db"):
                  os.remove("reasoner/train.db")
            if os.path.exists("reasoner/train_r.db"):
                  os.remove("reasoner/train_r.db")
            if os.path.exists("reasoner/trained.mln"):
                  os.remove("reasoner/trained.mln")
            if os.path.exists("reasoner/query.db"):
                  os.remove("reasoner/query.db")
            if os.path.exists("reasoner/query_r.db"):
                  os.remove("reasoner/query_r.db")
            if os.path.exists("reasoner/query.result"):
                  os.remove("reasoner/query.result")
            if os.path.exists("reasoner/query_r.result"):
                  os.remove("reasoner/query_r.result")    
            
            a_r,a_c,a_s=sim.minor_run(100,5)
            LPRA_r.append(a_r[0])
            LPRA_c.append(a_c[0])
            LPRA_s.append(a_s[0])
            PR_r.append(a_r[1])
            PR_c.append(a_c[1])
            PR_s.append(a_s[1])
            RA_r.append(a_r[2])
            RA_c.append(a_c[2])
            RA_s.append(a_s[2])
            A_r.append(a_r[3])
            A_c.append(a_c[3])
            A_s.append(a_s[3])
      
      mean_LPRA_r=sum(LPRA_r)/len(LPRA_r)
      mean_LPRA_c=sum(LPRA_c)/len(LPRA_c)
      mean_LPRA_s=sum(LPRA_s)/len(LPRA_s)
      mean_PR_r=sum(PR_r)/len(PR_r)
      mean_PR_c=sum(PR_c)/len(PR_c)
      mean_PR_s=sum(PR_s)/len(PR_s)
      mean_RA_r=sum(RA_r)/len(RA_r)
      mean_RA_c=sum(RA_c)/len(RA_c)
      mean_RA_s=sum(RA_s)/len(RA_s)
      mean_A_r=sum(A_r)/len(A_r)
      mean_A_c=sum(A_c)/len(A_c)
      mean_A_s=sum(A_s)/len(A_s)

      std_LPRA_r=st.stdev(LPRA_r)
      std_LPRA_c=st.stdev(LPRA_c)
      std_LPRA_s=st.stdev(LPRA_s)
      std_PR_r=st.stdev(PR_r)
      std_PR_c=st.stdev(PR_c)
      std_PR_s=st.stdev(PR_s)
      std_RA_r=st.stdev(RA_r)
      std_RA_c=st.stdev(RA_c)
      std_RA_s=st.stdev(RA_s)
      std_A_r=st.stdev(A_r)
      std_A_c=st.stdev(A_c)
      std_A_s=st.stdev(A_s)
      
      error_r=[std_LPRA_r,std_PR_r,std_RA_r,std_A_r]
      error_c=[std_LPRA_c,std_PR_c,std_RA_c,std_A_c]
      error_s=[std_LPRA_s,std_PR_s,std_RA_s,std_A_s]
      
      reward=[mean_LPRA_r,mean_PR_r,mean_RA_r,mean_A_r]
      cost=[mean_LPRA_c,mean_PR_c,mean_RA_c,mean_A_c]
      success=[mean_LPRA_s,mean_PR_s,mean_RA_s,mean_A_s]
      baselines=["LPRA","PR","RA","A"]

      print("LPRA:",LPRA_r,LPRA_c,LPRA_s)
      print("PR is:",PR_r,PR_c,PR_s)
      print("RA is:",RA_r,RA_c,RA_s)
      print("A is:",A_r,A_c,A_s)
      print(reward,cost,success)

      plt.figure()
      #plt.subplots()
      plt.bar(baselines,reward,yerr=error_r,capsize=2)
      plt.ylabel("Reward")
      plt.ylim(60,90)
      plt.savefig("a_r.pdf")
      plt.close()

      plt.figure()
      #plt.subplots()
      plt.bar(baselines,cost,yerr=error_c,capsize=2)
      plt.ylabel("Cost")
      plt.ylim(5,20)
      plt.savefig("a_c.pdf")

      plt.figure()
      #plt.subplots()
      plt.bar(baselines,success,yerr=error_s,capsize=2)
      plt.ylabel("Successrate")
      plt.ylim(0.7,1)
      plt.savefig("a_s.pdf")
      plt.close()
      """
      
      


      """
      index=0
      results=[]
      samples=[]
      instance1={"weather":"rainy","time_period":"busy","perception":"crowded","cr":"crowded","co":None}
      instance2={"weather":"rainy","time_period":"normal","perception":"empty","cr":"empty","co":None}
      instance3={"weather":"sunny","time_period":"normal","perception":"empty","cr":"empty","co":None}
      instance4={"weather":"sunny","time_period":"busy","perception":"crowded","cr":"crowded","co":None}
      f=open("reasoner/trial.db","w")
      Reasoner.minor_output_evd(f,instance1,"1")
      Reasoner.minor_output_evd(f,instance2,"2")
      Reasoner.minor_output_evd(f,instance3,"3")
      Reasoner.minor_output_evd(f,instance4,"4")
      f.close()
      step=200
      data_list=[]
      frequency_list=[]
      avg_list=[]
      reasoning_list=[]
      f_expected=[0.1,0.7,0.9,0.3]
      for i in range(1,20):
            training_data=sim.minor_generate_data(step)
            data_list=data_list+training_data
            fr,avg=sim.minor_check_co(data_list)
            print("Length is",len(data_list))
            frequency_list.append(fr)
            avg_list.append(avg)
            f=open("reasoner/train.db","a")
            for ins in training_data:
                  Reasoner.minor_output_data(f,ins,str(index))
                  index+=1
            f.close()
            Reasoner.learn_weights(input_file="autocar.mln",output_file="trained.mln",train_data="train.db")
            Reasoner.infer(result_file="test.result",evidence_file="trial.db")
            p_co_list=Reasoner.read_result("test.result")
            list_of_floats = [round(float(item),4) for item in p_co_list]

            result=list(np.absolute(np.array(list_of_floats)-np.array(f_expected)))           
            avg=sum(result)/len(result)
            reasoning_list.append(avg)

            results.append((round(float(p_co_list[0]),4)))
            samples.append(i*step)
            print(results,frequency_list,samples)
      plt.figure()
      plt.plot(samples,results,'.-b')
      plt.plot(samples,frequency_list,'.-r')
      plt.plot(samples,avg_list,'.-g')
      plt.plot(samples,reasoning_list,'.-y')
      plt.xlabel("Number of Samples")
      plt.ylabel("probability")

      plt.savefig("performance_of_MLN.pdf")
      plt.show()


      plt.close()
      """



                  
      


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
            
      

            