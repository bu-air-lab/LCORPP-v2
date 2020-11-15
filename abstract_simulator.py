# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:02:17 2020

@author: cckklt
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


import os
import shutil
import argparse
import json
import random
import math
import copy
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
from classifier.lidar_perception import perceive_learn
import perception
import reasoner
from interaction import Planner

random.seed(5)
np.random.seed(5)

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
      

      
      def minor_perceive(self,instance,conf_em,conf_cr):
            if instance['cr']=="empty":
                  perception=self.sample(self.crowded_list,conf_em)
            elif instance['cr']=="crowded":
                  perception=self.sample(self.crowded_list,conf_cr)

            return perception
      
      def minor_perceive_data(self,instances,conf_em,conf_cr):
            for ins in instances:
                  ins['perception']=self.minor_perceive(ins,conf_em,conf_cr)
      
      def minor_sample_cr(self,instance):
            if instance['time_period']=="busy":
                        cr=self.sample(self.crowded_list,[0.7,0.3])          
            elif instance["time_period"]=="normal":
                        cr=self.sample(self.crowded_list,[0.3,0.7])
            return cr
      
      def minor_sample_co(self,instance):
            if instance["weather"]=="rainy":
                  if instance['cr']=="crowded":
                        co=self.sample(self.willingness_list,[0.1,0.9])
                  elif instance['cr']=="empty":
                        co=self.sample(self.willingness_list,[0.2,0.8])
       
            elif instance["weather"]=="sunny":
                  if instance['cr']=="crowded":
                        co=self.sample(self.willingness_list,[0.8,0.2])  
                  elif instance['cr']=="empty":
                        co=self.sample(self.willingness_list,[0.9,0.1])
            return co
      
      def minor_create_instance(self):
            instance={}
            instance['weather']=random.choice(self.weather_list)
            instance['time_period']=random.choice(self.time_period_list)
            instance['cr']=self.minor_sample_cr(instance)
            instance['co']=self.minor_sample_co(instance)
            instance['perception']=None
            instance['cr_reasoning']=None
            instance['co_reasoning']=None
            instance['room_left']=random.choice(self.room_left_list)
            return instance
      
      def minor_generate_data(self,num_trials):
            instance_list=[]
            for i in range(0,num_trials):
                  instance=self.minor_create_instance()
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
            instance1={"time_period":"busy","weather":"rainy","perception":"crowded","cr":None,"co":None}
            instance2={"time_period":"busy","weather":"rainy","perception":"empty","cr":None,"co":None}
            instance3={"time_period":"busy","weather":"sunny","perception":"crowded","cr":None,"co":None}
            instance4={"time_period":"busy","weather":"sunny","perception":"empty","cr":None,"co":None}
            
            instance5={"time_period":"normal","weather":"rainy","perception":"crowded","cr":None,"co":None}
            instance6={"time_period":"normal","weather":"rainy","perception":"empty","cr":None,"co":None}
            instance7={"time_period":"normal","weather":"sunny","perception":"crowded","cr":None,"co":None}
            instance8={"time_period":"normal","weather":"sunny","perception":"empty","cr":None,"co":None}
            
            
            f=open("reasoner/query.db","w")
            reasoner.minor_output_evd(f,instance1,"1")
            reasoner.minor_output_evd(f,instance2,"2")
            reasoner.minor_output_evd(f,instance3,"3")
            reasoner.minor_output_evd(f,instance4,"4")
            reasoner.minor_output_evd(f,instance5,"5")
            reasoner.minor_output_evd(f,instance6,"6")
            reasoner.minor_output_evd(f,instance7,"7")
            reasoner.minor_output_evd(f,instance8,"8")
            f.close()

            f=open("reasoner/query_r.db","w")
            reasoner.minor_output_evd_r(f,instance1,"1")
            reasoner.minor_output_evd_r(f,instance3,"2")
            reasoner.minor_output_evd_r(f,instance5,"3")
            reasoner.minor_output_evd_r(f,instance7,"4")
            f.close()

            f=open("reasoner/query_cr.db","w")
            reasoner.minor_output_evd_cr(f,instance1,"1")
            reasoner.minor_output_evd_cr(f,instance2,"2")
            reasoner.minor_output_evd_cr(f,instance5,"3")
            reasoner.minor_output_evd_cr(f,instance6,"4")
            f.close()

            f=open("reasoner/query_r_cr.db","w")
            reasoner.minor_output_evd_r_cr(f,instance1,"1")
            reasoner.minor_output_evd_r_cr(f,instance2,"5")
            f.close()
      
      def minor_query_reasoner(self,q,instance_list):
            for ins in instance_list:
                  assert ins['perception']!=None,"perception is None"
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
                  assert ins['perception']!=None,"perception is None"
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
      
      def minor_create_testdata(self,num_data):
            test_list=[]
            #f=open("reasoner/test.db","w")
            for i in range(0,num_data):
                  instance=self.minor_create_instance()
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
                        ins["co_reasoning"]=0.9
                  elif ins['co']=='not cooperative':
                        ins["co_reasoning"]=0.1               
                  if ins['cr']=="empty":
                        ins["cr_reasoning"]=0.9
                  elif ins['cr']=='crowded':
                        ins["cr_reasoning"]=0.1               
                  print("Instance:",i,ins)
                  ins["result"]=planner.run(ins)
                  print("Instance:",i,ins['result'],ins['cost'],ins['reward'])
      
      def minor_a_cr(self,planner,instance_list):
            i=0
            for ins in instance_list:
                  i+=1
                  print("\n\nWith accurate congestion reasoning")
                  self.minor_initialize_planning(ins)
                  ins["co_reasoning"]=0.5              
                  
                  if ins['cr']=="empty":
                        #ins["cr_reasoning"]=self.sample([1,0],[0.9,0.1])
                        ins["cr_reasoning"]=0.9
                  elif ins['cr']=='crowded':
                        ins["cr_reasoning"]=0.1
                        #ins["cr_reasoning"]=self.sample([1,0],[0.1,0.9])              
                  print("Instance:",i,ins)
                  ins["result"]=planner.run(ins)
                  print("Instance:",i,ins['result'],ins['cost'],ins['reward'])
      
      def minor_a_co(self,planner,instance_list):
            i=0
            for ins in instance_list:
                  i+=1
                  print("\n\nWith accurate behavior reasoning")
                  self.minor_initialize_planning(ins)
                  ins["cr_reasoning"]=0.5              
                  
                  if ins['co']=="cooperative":
                        ins["co_reasoning"]=0.9
                  elif ins['co']=='not cooperative':
                        ins["co_reasoning"]=0.1               
                  print("Instance:",i,ins)
                  ins["result"]=planner.run(ins)
                  print("Instance:",i,ins['result'],ins['cost'],ins['reward'])
      
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
            
            if os.path.exists("reasoner/trained_r.mln"):
                  os.remove("reasoner/trained_r.mln")
      
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
      
      def minor_del_rlearn_file(self):
            if os.path.exists("reasoner/trained_rlearn.mln"):
                  os.remove("reasoner/trained_rlearn.mln")

            if os.path.exists("reasoner/train_rlearn.db"):
                  os.remove("reasoner/train_rlearn.db")
            
            if os.path.exists("reasoner/query_cr_rlearn.db"):
                  os.remove("reasoner/query_cr_rlearn.db")  
            
            if os.path.exists("reasoner/query_cr_rlearn.result"):
                  os.remove("reasoner/query_cr_rlearn.result")  

            if os.path.exists("reasoner/query_rlearn.db"):
                  os.remove("reasoner/query_rlearn.db")  
            
            if os.path.exists("reasoner/query_rlearn.result"):
                  os.remove("reasoner/query_rlearn.result")
      
      def minor_initialize_reasoning(self,instance_list):
            for ins in instance_list:
                  ins['cr_reasoning']=None
                  ins['co_reasoning']=None

      
      def minor_c_learning_only(self,test_data2,steps,conf):
            planner=Planner()
            c_learn_rewards=[]
            c_learn_successrates=[]
            conf_index2=0
            l=[]
            for  step in steps:
                  #p learning only                
                  for ins in test_data2:
                        ins['cr_reasoning']=0.5
                        if ins['co']=="cooperative":
                              ins['co_reasoning']=(conf[conf_index2][0][0])
                        elif ins['co']=='not cooperative':
                              ins['co_reasoning']=(conf[conf_index2][1][0])
                  
                  self.minor_planning(planner,test_data2)
                  c_learning_r,c_learning_c,c_learning_s=self.minor_get_metrics(test_data2)
                  
                  l.append(conf[conf_index2][0][0])
                  l.append(conf[conf_index2][1][0])
                  conf_index2+=1
                  c_learn_rewards.append(c_learning_r)
                  c_learn_successrates.append(c_learning_s)
            print(l)
            return c_learn_rewards,c_learn_successrates

      def minor_plearn_step(self,planner,test_data,conf_em,conf_cr):
            for ins in test_data:
                  ins['co_reasoning']=0.5
                  if ins['cr']=="empty":
                        #ins['cr_reasoning']=(conf_em[1])
                        ins['cr_reasoning']=self.sample([0,1],conf_em)
                  elif ins['cr']=='crowded':
                        #ins['cr_reasoning']=(conf_cr[1])
                        ins['cr_reasoning']=self.sample([0,1],conf_cr)
            #print("conf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",conf_em,conf_cr)
            #print("conf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",conf_em,conf_cr)
            self.minor_planning(planner,test_data)
            plearn_r,plearn_c,plearn_s=self.minor_get_metrics(test_data)
            #print("conf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",conf_em,conf_cr)
            #print("conf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",conf_em,conf_cr)
            return plearn_r,plearn_s
      
      def minor_rlearn_step(self,planner,training_data,test_data,index_rlearn):
            #perceive the test data
            self.minor_initialize_reasoning(test_data)
            self.minor_perceive_data(test_data,[0.5,0.5],[0.5,0.5])

            #generate training evidence for R learning only 
            f=open("reasoner/train_rlearn.db","a")
            for ins in training_data:
                  #ins['perception']=self.minor_perceive(ins,[0.5,0.5],[0.5,0.5])
                  reasoner.minor_output_data(f,ins,str(index_rlearn))
                  index_rlearn+=1
            f.close()

            #learn the weights from training data for R learning only
            reasoner.learn_weights(input_file="autocar.mln",output_file="trained_rlearn.mln",train_data="train_rlearn.db")
            reasoner.infer_cr(result_file="query_cr_rlearn.result",evidence_file="query_cr.db",mln_file="trained_rlearn.mln")
            reasoner.infer(result_file="query_rlearn.result",evidence_file="query.db",mln_file="trained_rlearn.mln")

            #get the query result for R learning only
            cr_query_rlearn=reasoner.read_result_cr("query_cr_rlearn.result")        
            co_query_rlearn=reasoner.read_result("query_rlearn.result")

            # map R learning only results to test data       
            self.minor_query_congestion(cr_query_rlearn,test_data)
            self.minor_query_reasoner(co_query_rlearn,test_data)
            self.minor_planning(planner,test_data)
            rlearn_r,rlearn_c,rlearn_s=self.minor_get_metrics(test_data)
            return rlearn_r,rlearn_s,index_rlearn



      def minor_run(self,test_data,steps):
            self.minor_del_reasoner_file()
            self.minor_del_rlearn_file()
            perception.delete_train_images()
            
            cr_expect=[]
            co_expect=[]
            test_data=test_data
            test_data2=copy.deepcopy(test_data)
            cr_test=self.minor_check_cr(test_data)
            co_test=self.minor_check_co(test_data)
            
            data_list=[]
            index=0
            index_r=0
            index_rlearn=0
            planner=Planner()
            lpra_r=[]
            lpra_c=[]
            lpra_s=[]
            a_r=[]
            a_c=[]
            a_s=[]
            
            p_learn_rewards=[]
            p_learn_successrates=[]
            r_learn_rewards=[]
            r_learn_successrates=[]

            samples=[]
            cr_diffs=[]
            co_diffs=[]
            conf_index=0
            conf_index2=0
            self.minor_generate_query_evidence()

            for step in steps:
                  #generate training data
                  training_data=random.choices(test_data,k=step)
                  data_list=data_list+training_data
                  print("Length is",len(data_list))
                  samples.append(len(data_list))
                  
                  perception.select_train_images(training_data)
                  perception_conf=list(perceive_learn())
                  conf_em=[1-perception_conf[0],perception_conf[0]]
                  conf_cr=[perception_conf[1],1-perception_conf[1]]
                  
                  #save conf_matrix to check
                  f=open("conf_matrix.txt","a")
                  f.write(str(perception_conf)+"\n")
                  f.write(str(conf_em)+"\n")
                  f.write(str(conf_cr)+"\n")
                  f.write("\n")
                  f.close()

                  #generate training evidences for LPRA
                  f=open("reasoner/train.db","a")
                  for ins in training_data:
                        #perception
                        ins['perception']=self.minor_perceive(ins,conf_em,conf_cr)
                        reasoner.minor_output_data(f,ins,str(index))
                        index+=1
                  f.close()

                  #generate training evidence for RA
                  f=open("reasoner/train_r.db","a")
                  for ins in training_data:
                        reasoner.minor_output_data_r(f,ins,str(index_r))
                        index_r+=1
                  f.close()


                  #learn the weights from training data for LPRA
                  reasoner.learn_weights(input_file="autocar.mln",output_file="trained.mln",train_data="train.db")
                  reasoner.infer_cr(result_file="query_cr.result",evidence_file="query_cr.db",mln_file="trained.mln")
                  reasoner.infer(result_file="query.result",evidence_file="query.db",mln_file="trained.mln")
                  #get the query results for LPRA
                  co_query=reasoner.read_result("query.result")
                  cr_query=reasoner.read_result_cr("query_cr.result")

                  #learn the weights from training data for RA
                  reasoner.learn_weights(input_file="autocar_r.mln",output_file="trained_r.mln",train_data="train_r.db")
                  reasoner.infer_cr(result_file="query_r_cr.result",evidence_file="query_r_cr.db",mln_file="trained_r.mln")
                  reasoner.infer(result_file="query_r.result",evidence_file="query_r.db",mln_file="trained_r.mln")
                  #get the query resultf for RA
                  co_query_r=reasoner.read_result("query_r.result")
                  cr_query_r=reasoner.read_result_cr("query_r_cr.result")

                  #self.minor_check_reasoner(training_data,cr_expect,co_expect,cr_query,co_query)
                  #check the reasoner learning result
                  print(cr_query,cr_test)
                  cr_diff=self.minor_check_prob(cr_query,cr_test)
                  co_diff=self.minor_check_prob(co_query,co_test)
                  co_diffs.append(co_diff)
                  cr_diffs.append(cr_diff)
                  
                  # map the LPRA reasoning results to test data
                  self.minor_perceive_data(test_data,conf_em,conf_cr)
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
                  #self.minor_perceive_data(test_data,conf[conf_index][0],conf[conf_index][1])
                  self.minor_query_congestion_r(cr_query_r,test_data)
                  self.minor_query_reasoner_r(co_query_r,test_data)
                  self.minor_planning(planner,test_data)
                  self.save_data(test_data,"RA")
                  avg_r_ra,avg_c_ra,avg_s_ra=self.minor_get_metrics(test_data)

                  plearn_r,plearn_s=self.minor_plearn_step(planner,test_data2,conf_em,conf_cr)
                  p_learn_rewards.append(plearn_r)
                  p_learn_successrates.append(plearn_s)
                  self.save_data(test_data,"plearn")

                  rlearn_r,rlearn_s,index_rlearn=self.minor_rlearn_step(planner,training_data,test_data2,index_rlearn)
                  r_learn_rewards.append(rlearn_r)
                  r_learn_successrates.append(rlearn_s)
                  self.save_data(test_data,"rlearn")
                  #conf_index+=1
            
            #apply only A on test data
            self.minor_a(planner,test_data)
            self.save_data(test_data,"A")
            avg_r_a,avg_c_a,avg_s_a=self.minor_get_metrics(test_data) 

            a_r.append(avg_r)
            a_c.append(avg_c)
            a_s.append(avg_s)

            a_r.append(avg_r_pr)
            a_c.append(avg_c_pr)
            a_s.append(avg_s_pr)
            
            a_r.append(avg_r_ra)
            a_c.append(avg_c_ra)
            a_s.append(avg_s_ra)

            a_r.append(avg_r_a)
            a_c.append(avg_c_a)
            a_s.append(avg_s_a)

            #plot the learning trend for LPRA   
            print("Samples is",samples)    
            self.minor_plot_trend(samples,lpra_r,"Training Data Size","Average Reward on Test Data",str(datetime.datetime.now())+"reward.pdf")
            self.minor_plot_trend(samples,lpra_c,"Training Data Size","Average Cost on Test Data",str(datetime.datetime.now())+"cost.pdf")
            self.minor_plot_trend(samples,lpra_s,"Training Data Size","Success rate on Test Data",str(datetime.datetime.now())+"success_rate.pdf")
            self.minor_plot_trend(samples,co_diffs,"Training Data Size","avergae prediction differences","co_trend.pdf")
            self.minor_plot_trend(samples,cr_diffs,"Training Data Size","avergae prediction differences","cr_trend.pdf")

        
            print("average is",a_r,a_c,a_s)

            baselines=["PERIL","R+A-","R+A","A"]
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
            print("cr_query",cr_query)
            return a_r,a_c,a_s,lpra_r,lpra_c,lpra_s,samples,r_learn_rewards,r_learn_successrates,p_learn_rewards,p_learn_successrates
      
      
      def minor_multi_run(self,test_data,steps):
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
            trends_r_learn_r=[]
            trends_r_learn_s=[]
            trends_p_learn_r=[]
            trends_p_learn_s=[]
            trends_c_learn_r=[]
            trends_c_learn_s=[]
            test_data2=copy.deepcopy(test_data)
            test_data3=copy.deepcopy(test_data)
            
            for i in range(5):
                  #self.minor_del_reasoner_file()       
                  a_r,a_c,a_s,l_trend_r,l_trend_c,l_trend_s,datasizes,rlearn_trend_r,rlearn_trend_s,p_learn_rewards,p_learn_successrates=self.minor_run(test_data,steps)
                  
                  trends_r.append(l_trend_r)
                  trends_c.append(l_trend_c)
                  trends_s.append(l_trend_s)
                  trends_r_learn_r.append(rlearn_trend_r)
                  trends_r_learn_s.append(rlearn_trend_s)
                  trends_p_learn_r.append(p_learn_rewards)
                  trends_p_learn_s.append(p_learn_successrates)

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
            err_r_learn_rewards=[[trends_r_learn_r[x][y] for x in range(len(trends_r_learn_r))] for y in range(batch_num-1)]
            err_r_learn_success=[[trends_r_learn_s[x][y] for x in range(len(trends_r_learn_s))] for y in range(batch_num-1)]
            err_p_learn_rewards=[[trends_p_learn_r[x][y] for x in range(len(trends_p_learn_r))] for y in range(batch_num-1)]
            err_p_learn_success=[[trends_p_learn_s[x][y] for x in range(len(trends_p_learn_s))] for y in range(batch_num-1)]
            #err_c_learn_rewards=[[trends_p_learn_r[x][y] for x in range(len(trends_c_learn_r))] for y in range(batch_num-1)]
            #err_c_learn_success=[[trends_p_learn_s[x][y] for x in range(len(trends_c_learn_s))] for y in range(batch_num-1)]


            mean_trend_r=[sum(err_r[x])/len(err_r[x]) for x in range(batch_num-1)]
            mean_trend_c=[sum(err_c[x])/len(err_c[x]) for x in range(batch_num-1)]
            mean_trend_s=[sum(err_s[x])/len(err_s[x]) for x in range(batch_num-1)]
            mean_trend_r_learn_rewards=[sum(err_r_learn_rewards[x])/len(err_r_learn_rewards[x]) for x in range(batch_num-1)]
            mean_trend_r_learn_success=[sum(err_r_learn_success[x])/len(err_r_learn_success[x]) for x in range(batch_num-1)]
            mean_trend_p_learn_rewards=[sum(err_p_learn_rewards[x])/len(err_p_learn_rewards[x]) for x in range(batch_num-1)]
            mean_trend_p_learn_success=[sum(err_p_learn_success[x])/len(err_p_learn_success[x]) for x in range(batch_num-1)]
            #mean_trend_c_learn_rewards=[sum(err_c_learn_rewards[x])/len(err_c_learn_rewards[x]) for x in range(batch_num-1)]
            #mean_trend_c_learn_success=[sum(err_c_learn_success[x])/len(err_c_learn_success[x]) for x in range(batch_num-1)]
            

            std_trend_r=[st.stdev(err_r[x]) for x in range(batch_num-1)]
            std_trend_c=[st.stdev(err_c[x]) for x in range(batch_num-1)]
            std_trend_s=[st.stdev(err_s[x]) for x in range(batch_num-1)]
            
            std_trend_r_learn_rewards=[st.stdev(err_r_learn_rewards[x]) for x in range(batch_num-1)]
            std_trend_r_learn_success=[st.stdev(err_r_learn_success[x]) for x in range(batch_num-1)]
            std_trend_p_learn_rewards=[st.stdev(err_p_learn_rewards[x]) for x in range(batch_num-1)]
            std_trend_p_learn_success=[st.stdev(err_p_learn_success[x]) for x in range(batch_num-1)]
            #std_trend_c_learn_rewards=[st.stdev(err_c_learn_rewards[x]) for x in range(batch_num-1)]
            #std_trend_c_learn_success=[st.stdev(err_c_learn_success[x]) for x in range(batch_num-1)]

            #datasizes=[10,20,40,80,160]
            plt.figure()
            #plt.subplots()
            plt.errorbar(datasizes,mean_trend_r,yerr=std_trend_r,capsize=2,label="LPRA Learning")
            plt.errorbar(datasizes,mean_trend_r_learn_rewards,yerr=std_trend_r_learn_rewards,capsize=2,ls='--',label="Reasoner Learning only")
            plt.errorbar(datasizes,mean_trend_p_learn_rewards,yerr=std_trend_p_learn_rewards,capsize=2,ls='--',label="Perception Learning only")
            #plt.errorbar(datasizes,mean_trend_c_learn_rewards,yerr=std_trend_c_learn_rewards,capsize=2,label="Cooperative Learning only")
            plt.xscale('log',base=2)
            plt.ylabel("Reward")
            plt.xlabel("Size of Data")
            #plt.ylim(30,)
            plt.legend()
            plt.savefig("LPRA_reward.pdf")
            plt.close()

            plt.figure()
            #plt.subplots()
            plt.errorbar(datasizes,mean_trend_c,yerr=std_trend_c,capsize=2)
            plt.xscale('log',base=2)
            plt.xlabel("Size of Data")
            plt.ylabel("Cost")
            #plt.ylim(50,80)
            plt.savefig("LPRA_cost.pdf")
            plt.close()

            plt.figure()
            #plt.subplots()
            plt.errorbar(datasizes,mean_trend_s,yerr=std_trend_s,capsize=2,label="LPRA Learning")
            plt.errorbar(datasizes,mean_trend_r_learn_success,yerr=std_trend_s,capsize=2,label="Reasoner Learning only")
            plt.errorbar(datasizes,mean_trend_p_learn_success,yerr=std_trend_s,capsize=2,label="Perception Learning only")
            plt.xscale('log',base=2)
            plt.xlabel("Size of Data")
            plt.ylabel("Success rate")
            #plt.ylim(0,6,)
            plt.legend()
            plt.savefig("LPRA_successrate.pdf")
            plt.close()

            y1_rewards=list(np.array(mean_trend_r)+np.array(std_trend_r))
            y2_rewards=list(np.array(mean_trend_r)-np.array(std_trend_r))
            y1_rlearn_rewards=list(np.array(mean_trend_r_learn_rewards)+np.array(std_trend_r_learn_rewards))
            y2_rlearn_rewards=list(np.array(mean_trend_r_learn_rewards)-np.array(std_trend_r_learn_rewards))
            y1_plearn_rewards=list(np.array(mean_trend_p_learn_rewards)+np.array(std_trend_p_learn_rewards))
            y2_plearn_rewards=list(np.array(mean_trend_p_learn_rewards)-np.array(std_trend_p_learn_rewards))


            plt.figure()
            #plt.subplots()
            plt.plot(datasizes,mean_trend_r,label="LPRA Learning")
            plt.fill_between(datasizes,y1_rewards,y2_rewards,alpha=0.5)
            
            plt.plot(datasizes,mean_trend_r_learn_rewards,ls='--',label="Reasoner Learning only")
            plt.fill_between(datasizes,y1_rlearn_rewards,y2_rlearn_rewards,alpha=0.5)
            
            plt.plot(datasizes,mean_trend_p_learn_rewards,ls='--',label="Perception Learning only")
            plt.fill_between(datasizes,y1_plearn_rewards,y2_plearn_rewards,alpha=0.5)
            #plt.errorbar(datasizes,mean_trend_c_learn_rewards,yerr=std_trend_c_learn_rewards,capsize=2,label="Cooperative Learning only")
            plt.ylabel("Reward")
            plt.xlabel("Size of Data")
            plt.xscale('log',basex=2)
            #plt.ylim(30,)
            plt.legend(loc= 'lower right')
            plt.savefig("LPRA_reward_bands.pdf")
            plt.close()

            """
            plt.figure()
            #plt.subplots()
            plt.plot(datasizes,mean_trend_s,yerr=std_trend_s,capsize=2,label="LPRA Learning")
            plt.plot(datasizes,mean_trend_r_learn_success,yerr=std_trend_s,capsize=2,label="Reasoner Learning only")
            plt.plot(datasizes,mean_trend_p_learn_success,yerr=std_trend_s,capsize=2,label="Perception Learning only")
            plt.xlabel("Datasize($e^x$)")
            plt.ylabel("Success rate")
            #plt.ylim(0,6,)
            plt.legend()
            plt.savefig("LPRA_successrate_bands.pdf")
            plt.close()
            """

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
            #plt.ylim(35,)
            plt.savefig("reward_comparison.pdf")
            plt.close()

            plt.figure()
            #plt.subplots()
            plt.bar(baselines,cost,yerr=error_c,capsize=2,width=0.6)
            plt.ylabel("Cost")
            #plt.ylim(5,)
            plt.savefig("cost_comparison.pdf")

            plt.figure()
            #plt.subplots()
            plt.bar(baselines,success,yerr=error_s,capsize=2,width=0.6)
            plt.ylabel("Successrate")
            plt.ylim(0.6,)
            plt.savefig("success_comparison.pdf")
            plt.close()


def main():
      parser = argparse.ArgumentParser(description='Training Settings')
      #parser.add_argument()
      sim=AbstractSimulator()  
      steps=[25,25,25,25,50,50,100,100,100]
      test_data=sim.minor_create_testdata(1800)
      #sim.minor_multi_run(test_data,steps)
      sim.minor_run(test_data,steps)
      

if __name__ == '__main__':
	main()
            
      

            