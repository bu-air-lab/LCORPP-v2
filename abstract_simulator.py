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
import pandas as pd
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
import dtio
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
                        co=self.sample(self.willingness_list,[0.05,0.95])
                  elif instance['cr']=="empty":
                        co=self.sample(self.willingness_list,[0.1,0.9])
       
            elif instance["weather"]=="sunny":
                  if instance['cr']=="crowded":
                        co=self.sample(self.willingness_list,[0.9,0.1])  
                  elif instance['cr']=="empty":
                        co=self.sample(self.willingness_list,[0.95,0.05])
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
                  instance['name']=str(i)
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
                  
                  elif ins['weather']=="sunny" and ins['time_period']=="normal" and ins['perception']=="crowded":
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
            p6=r_n_em_co/(r_n_em+1)
            p7=s_n_cr_co/(s_n_cr+1)
            p8=s_n_em_co/(s_n_em+1)

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
            reasoner.minor_output_evd_r(f,instance3,"3")
            reasoner.minor_output_evd_r(f,instance5,"5")
            reasoner.minor_output_evd_r(f,instance7,"7")
            f.close()

            f=open("reasoner/query_cr.db","w")
            reasoner.minor_output_evd_cr(f,instance1,"1")
            reasoner.minor_output_evd_cr(f,instance2,"2")
            reasoner.minor_output_evd_cr(f,instance5,"5")
            reasoner.minor_output_evd_cr(f,instance6,"6")
            f.close()

            f=open("reasoner/query_r_cr.db","w")
            reasoner.minor_output_evd_r_cr(f,instance1,"1")
            reasoner.minor_output_evd_r_cr(f,instance5,"5")
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
            metrics_dict={}
            reward=0
            cost=0
            success=0
            left_signals=0
            move_lefts=0
            for ins in ins_list:
                  reward+=ins['reward']
                  cost+=ins['cost']
                  if ins['result']=="Successful":
                        success+=1
                  left_signals+=ins['actions'][0]
                  move_lefts+=ins['actions'][0]
            l=len(ins_list)
            metrics_dict['avg_signals']=left_signals/l
            metrics_dict['avg_moves']=move_lefts/l
            metrics_dict['avg_reward']=reward/l
            metrics_dict['avg_cost']=cost/l
            metrics_dict['success_rate']=success/l
            return metrics_dict
      
      def save_data(self,instances,name):
            f=open(name+".json","w")
            for ins in instances:
                  if ins['result']=="Failed":
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

      def minor_initialize_reasoning(self,instance_list):
            for ins in instance_list:
                  ins['cr_reasoning']=None
                  ins['co_reasoning']=None
      
      def dataszies_to_steps(self,datasizes):
            steps=[]
            for i in range(len(datasizes)):
                  if i==0:
                        steps.append(datasizes[i])
                  else:
                        steps.append(datasizes[i]-datasizes[i-1])
            return steps

      def minor_plearn_step(self,planner,test_data,conf_em,conf_cr):
            f=open("conf_matrix.txt","a")
            #f.write(str(perception_conf)+"\n")
            f.write(str(conf_em)+"\n")
            f.write(str(conf_cr)+"\n")
            f.write("\n")
            f.close()
            for ins in test_data:
                  ins['co_reasoning']=0.5
                  if ins['cr']=="empty":
                        #ins['cr_reasoning']=(conf_em[1])
                        ins['cr_reasoning']=self.sample([0,1],conf_em)
                  elif ins['cr']=='crowded':
                        #ins['cr_reasoning']=(conf_cr[1])
                        ins['cr_reasoning']=self.sample([0,1],conf_cr)
            self.minor_planning(planner,test_data)
            plearn_metrics=self.minor_get_metrics(test_data)
            #print("conf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",conf_em,conf_cr)
            return plearn_metrics

      def minor_rlearn_step(self,planner,training_data,data_collected,test_data):
            #perceive the test data
            self.minor_initialize_reasoning(test_data)
            self.minor_perceive_data(test_data,[0.5,0.5],[0.5,0.5])

            #generate training evidence for R learning only 
            f=open("reasoner/train_rlearn.db","a")
            for ins in training_data:
                  ins['perception']=self.minor_perceive(ins,[0.5,0.5],[0.5,0.5])
                  reasoner.minor_output_data(f,ins,ins['name'])
            f.close()

            #learn the weights from training data for R learning only
            reasoner.learn_weights(input_file="autocar.mln",output_file="trained_rlearn.mln",train_data="train_rlearn.db")
            reasoner.infer_cr(result_file="query_cr_rlearn.result",evidence_file="query_cr.db",mln_file="trained_rlearn.mln")
            reasoner.infer(result_file="query_rlearn.result",evidence_file="query.db",mln_file="trained_rlearn.mln")

            #get the query result for R learning only
            cr_query_rlearn=reasoner.read_result_cr("query_cr_rlearn.result")        
            co_query_rlearn=reasoner.read_result("query_rlearn.result")
            
            cr_samples_count=self.minor_check_cr(data_collected)
            cr_test_count=self.minor_check_cr(test_data)
            co_samples_count=self.minor_check_co(data_collected)
            co_test_count=self.minor_check_co(test_data)

            dtio.write_row("rlearn_cr.csv",cr_query_rlearn)
            dtio.write_row("rlearn_cr.csv",cr_samples_count)
            dtio.write_row("rlearn_cr.csv",cr_test_count)
            dtio.write_row("rlearn_co.csv",co_query_rlearn)
            dtio.write_row("rlearn_co.csv",co_samples_count)
            dtio.write_row("rlearn_co.csv",co_test_count)

            # map R learning only results to test data       
            self.minor_query_congestion(cr_query_rlearn,test_data)
            self.minor_query_reasoner(co_query_rlearn,test_data)
            self.minor_planning(planner,test_data)
            rlearn_metrics=self.minor_get_metrics(test_data)
            return rlearn_metrics
      
      def minor_peril_step(self,planner,training_data,data_collected,test_data,conf_em,conf_cr):
            self.minor_initialize_reasoning(test_data)
            #generate training evidences for LPRA
            f=open("reasoner/train.db","a")
            for ins in training_data:
                  #perception
                  ins['perception']=self.minor_perceive(ins,conf_em,conf_cr)
                  reasoner.minor_output_data(f,ins,ins['name'])
            f.close()
            #learn the weights from training data for LPRA
            reasoner.learn_weights(input_file="autocar.mln",output_file="trained.mln",train_data="train.db")
            reasoner.infer_cr(result_file="query_cr.result",evidence_file="query_cr.db",mln_file="trained.mln")
            reasoner.infer(result_file="query.result",evidence_file="query.db",mln_file="trained.mln")
            #get the query results for LPRA
            co_query=reasoner.read_result("query.result")
            cr_query=reasoner.read_result_cr("query_cr.result")
            #check the reasoner learning result

            # map the LPRA reasoning results to test data
            self.minor_perceive_data(test_data,conf_em,conf_cr)
            self.minor_query_reasoner(co_query,test_data)
            self.minor_query_congestion(cr_query,test_data)
            self.minor_planning(planner,test_data)
            self.save_data(test_data,"PERIL")
            peril_metrics=self.minor_get_metrics(test_data)
            return peril_metrics
      
      def minor_run(self,test_data,steps):
            reasoner.minor_del_reasoner_file()
            reasoner.minor_del_rlearn_file()
            perception.delete_train_images()
            
            cr_expect=[]
            co_expect=[]
            test_data=test_data
            test_data2=copy.deepcopy(test_data)
            
            data_collected=[]
            planner=Planner()
            peril_rewards=[]
            peril_costs=[]
            peril_successrates=[]
            
            plearn_rewards=[]
            plearn_costs=[]
            plearn_successrates=[]

            rlearn_rewards=[]
            rlearn_costs=[]
            rlearn_successrates=[]

            samples=[]
            cr_diffs=[]
            co_diffs=[]
            self.minor_generate_query_evidence()

            for step in steps:
                  #generate training data
                  data_source=[x for x in test_data if x not in data_collected]
                  training_data=random.sample(data_source,k=step)
                  data_collected=data_collected+training_data
                  print("Length is",len(data_collected))
                  samples.append(len(data_collected))
                  
                  perception.select_train_images(training_data)
                  perception_conf=list(perceive_learn())
                  conf_em=[1-perception_conf[0],perception_conf[0]]
                  conf_cr=[perception_conf[1],1-perception_conf[1]]
                  #save conf_matrix to check

                  peril_metrics=self.minor_peril_step(planner,training_data,data_collected,test_data,conf_em,conf_cr)
                  peril_rewards.append(peril_metrics['avg_reward'])
                  peril_costs.append(peril_metrics['avg_cost'])
                  peril_successrates.append(peril_metrics['success_rate'])

                  # map the PR results to test data
                  self.minor_planning_pr(planner,test_data)
                  self.save_data(test_data,"PRA_")
                  pr_metrics=self.minor_get_metrics(test_data)

                  plearn_metrics=self.minor_plearn_step(planner,test_data2,conf_em,conf_cr)
                  plearn_rewards.append(plearn_metrics['avg_reward'])
                  plearn_costs.append(plearn_metrics['avg_cost'])
                  plearn_successrates.append(plearn_metrics['success_rate'])

                  rlearn_metrics=self.minor_rlearn_step(planner,training_data,data_collected,test_data2)
                  rlearn_rewards.append(rlearn_metrics['avg_reward'])
                  rlearn_costs.append(rlearn_metrics['avg_cost'])
                  rlearn_successrates.append(rlearn_metrics['success_rate'])
            
            #apply only A on test data
            self.minor_a(planner,test_data)
            self.save_data(test_data,"A")
            #avg_r_a,avg_c_a,avg_s_a=self.minor_get_metrics(test_data)
            a_metrics=self.minor_get_metrics(test_data)

            reward_comparison=[peril_metrics['avg_reward'],pr_metrics['avg_reward'],rlearn_metrics['avg_reward'],a_metrics['avg_reward']]
            cost_comparison=[peril_metrics['avg_cost'],pr_metrics['avg_cost'],rlearn_metrics['avg_cost'],a_metrics['avg_cost']]
            successrate_comparison=[peril_metrics['success_rate'],pr_metrics['success_rate'],rlearn_metrics['success_rate'],a_metrics['success_rate']]
            
            plearn_mat=[plearn_rewards,plearn_costs,plearn_successrates]
            rlearn_mat=[rlearn_rewards,rlearn_costs,rlearn_successrates]
            peril_mat=[peril_rewards,peril_costs,peril_successrates]
            return samples,reward_comparison,cost_comparison,successrate_comparison,peril_mat,rlearn_mat,plearn_mat
      
      def minor_multi_run(self,test_data,steps):
            trends_r=[]
            trends_c=[]
            trends_s=[]
            
            trends_rlearn_r=[]
            trends_rlearn_c=[]
            trends_rlearn_s=[]
            trends_plearn_r=[]
            trends_plearn_c=[]
            trends_plearn_s=[]

            reward_comps=[]
            cost_comps=[]
            successrate_comps=[]
            dtio.clear_results_data()
            
            for i in range(5):    
                  datasizes,reward_4b,cost_4b,successrate_4b,peril_mat,rlearn_mat,plearn_mat=self.minor_run(test_data,steps)
                  
                  trends_r.append(peril_mat[0])
                  trends_c.append(peril_mat[1])
                  trends_s.append(peril_mat[2])
                  
                  trends_rlearn_r.append(rlearn_mat[0])
                  trends_rlearn_c.append(rlearn_mat[1])
                  trends_rlearn_s.append(rlearn_mat[2])
                  
                  trends_plearn_r.append(plearn_mat[0])
                  trends_plearn_c.append(plearn_mat[1])
                  trends_plearn_s.append(plearn_mat[2])

                  reward_comps.append(reward_4b)
                  cost_comps.append(cost_4b)
                  successrate_comps.append(successrate_4b)

            #row(batch)*column(run)
            dtio.write_csv("peril_rewards.csv",trends_r)
            dtio.write_csv("peril_costs.csv",trends_c)
            dtio.write_csv("peril_successrates.csv",trends_s)

            dtio.write_csv("rlearn_rewards.csv",trends_rlearn_r)
            dtio.write_csv("rlearn_costs.csv",trends_rlearn_c)
            dtio.write_csv("rlearn_successrates.csv",trends_rlearn_s)

            dtio.write_csv("plearn_rewards.csv",trends_plearn_r)
            dtio.write_csv("plearn_costs.csv",trends_plearn_c)
            dtio.write_csv("plearn_successrates.csv",trends_plearn_s)

            #row(run)*colomn(baselines)
            dtio.write_csv("reward_comps.csv",reward_comps)
            dtio.write_csv("cost_comps.csv",cost_comps)
            dtio.write_csv("successrate_comps.csv",successrate_comps)
            
            dtio.plot_learning(datasizes)
            dtio.plot_learning_errband(datasizes)
            dtio.plot_comparison()


def main():
      parser = argparse.ArgumentParser(description='Training Settings')
      #parser.add_argument()
      sim=AbstractSimulator()  
      #steps=[30,60,60,60,60,60,60]
      steps=sim.dataszies_to_steps([30,90,150,210,270,330,390])
      test_data=sim.minor_generate_data(4000)
      sim.minor_multi_run(test_data,steps)
      #sim.minor_run(test_data,steps)
      

if __name__ == '__main__':
	main()
            
      
