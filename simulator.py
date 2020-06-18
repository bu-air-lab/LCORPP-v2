from utils.parser import Policy,Solver
from utils.pomdp_parser import Model
import numpy as np
np.set_printoptions(precision=2)     # for better belief printing 
import random
import pathlib
import pandas as pd


class Simulator:

	def __init__(self):

		pomdp_file = pathlib.Path.cwd()/'model/program.pomdp'
		policy_file = pathlib.Path.cwd()/'model/program.policy'

		self.model = Model(pomdp_file=pomdp_file, parsing_print_flag=False)
		self.policy = Policy(len(self.model.states),
							len(self.model.actions),
							policy_file=policy_file)

	def update(self, a_idx,o_idx,b ):
 		'''Update belief using Bayes update rule'''
		b = np.dot(b, self.model.trans_mat[a_idx, :])
		b = [b[i] * self.model.obs_mat[a_idx, i, o_idx] for i in range(len(self.model.states))]
		b = b / sum(b)
 
		return b

	#def pretty_print(self,b):
		#df = pd.DataFrame(b,index=False, columns=self.model.states)
		#print (df)

	def observe(self,a_idx):
		'''Make an observation using random distribution of the observation marix'''
		all_obs_idx= range(len(self.model.observations))
		prob_dist = self.model.obs_mat[a_idx,random.randint(0,len(self.model.states)-1),:]
		
		return np.random.choice(all_obs_idx, p= prob_dist) 

	def run(self):

		#Initialize belief
		b = np.ones(len(self.model.states))/(len(self.model.states)-1)
		b[-1]=0.0

		print (b)
		
		term=False

		while not term:
			a_idx=self.policy.select_action(b)
			obs_idx = self.observe(a_idx)
			print ('\n\n\naction is: ',self.model.actions[a_idx])
			print ('observation is: ',self.model.observations[obs_idx]) 
			b = self.update(a_idx,obs_idx,b)
			print(b)
			
			if b[-1]>0:
				term=True
				print('\n')

def main():
	instance= Simulator()
	instance.run()



if __name__ == '__main__':
	main()