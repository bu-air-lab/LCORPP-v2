from abstract_simulator import AbstractSimulator
from interaction import Planner
import matplotlib.pyplot as plt
import statistics as st
import dtio
import reasoner

def minor_a_gt(sim,planner,instance_list):
	i=0
	for ins in instance_list:
		i+=1
		print("\n\nIdeal Peril")
		sim.minor_initialize_planning(ins)
		
		if ins['cr']=="empty":
			ins["cr_reasoning"]=sim.sample([0.9,0.1],[0.9,0.1])
			#ins["cr_reasoning"]=0.9
		elif ins['cr']=='crowded':
			#ins["cr_reasoning"]=0.1
			ins["cr_reasoning"]=sim.sample([0.9,0.1],[0.1,0.9])
		reasoner.reason_prob_co(ins)

		print("Instance:",i,ins)
		ins["result"]=planner.run(ins)
		print("Instance:",i,ins['result'],ins['cost'],ins['reward'])

def minor_a_cr(sim,planner,instance_list):
	i=0
	for ins in instance_list:
		i+=1
		print("\n\nPerception only")
		sim.minor_initialize_planning(ins)
		ins["co_reasoning"]=0.5
		if ins['cr']=="empty":
			ins["cr_reasoning"]=sim.sample([1,0],[0.9,0.1])
			#ins["cr_reasoning"]=0.9
		elif ins['cr']=='crowded':
			#ins["cr_reasoning"]=0.1
			ins["cr_reasoning"]=sim.sample([1,0],[0.1,0.9])
		print("Instance:",i,ins)
		ins["result"]=planner.run(ins)
		print("Instance:",i,ins['result'],ins['cost'],ins['reward'])

def minor_a_co(sim,planner,instance_list):
	i=0
	for ins in instance_list:
		i+=1
		print("\n\nReason only")
		sim.minor_initialize_planning(ins)
		ins["cr_reasoning"]=0.5
		reasoner.reason_prob_co(ins)
		print("Instance:",i,ins)
		ins["result"]=planner.run(ins)
		print("Instance:",i,ins['result'],ins['cost'],ins['reward'])

def minor_a_test(self,instance_list):
	planner=Planner()
	a_r=[]
	a_c=[]
	a_s=[]
	self.minor_a(planner,instance_list)
	#self.save_data(instance_list,"A")
	avg_r_a,avg_c_a,avg_s_a=self.minor_get_metrics(instance_list)
	a_r.append(avg_r_a)
	a_c.append(avg_c_a)
	a_s.append(avg_s_a)

	self.minor_a_gt(planner,instance_list)
	#self.save_data(instance_list,"A_gt")
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


def minor_a_test_err(sim,instance_list):
	planner=Planner()
	a_r=[]
	a_c=[]
	a_s=[]
	a_r_gt=[]
	a_c_gt=[]
	a_s_gt=[]
	a_r_cr=[]
	a_c_cr=[]
	a_s_cr=[]
	a_r_co=[]
	a_c_co=[]
	a_s_co=[]

	for i in range(2):
		sim.minor_a(planner,instance_list)
		#sim.save_data(instance_list,"A")
		a_metrics=sim.minor_get_metrics(instance_list)
		a_r.append(a_metrics['avg_reward'])
		a_c.append(a_metrics['avg_cost'])
		a_s.append(a_metrics['success_rate'])

		minor_a_gt(sim,planner,instance_list)
		#sim.save_data(instance_list,"A_0.9")
		peril_metrics=sim.minor_get_metrics(instance_list)
		a_r_gt.append(peril_metrics['avg_reward'])
		a_c_gt.append(peril_metrics['avg_cost'])
		a_s_gt.append(peril_metrics['success_rate'])

		minor_a_cr(sim,planner,instance_list)
		#sim.save_data(instance_list,"A_cr")
		cr_metrics=sim.minor_get_metrics(instance_list)
		a_r_cr.append(cr_metrics['avg_reward'])
		a_c_cr.append(cr_metrics['avg_cost'])
		a_s_cr.append(cr_metrics['success_rate'])

		minor_a_co(sim,planner,instance_list)
		#sim.save_data(instance_list,"A_co")
		co_metrics=sim.minor_get_metrics(instance_list)
		a_r_co.append(co_metrics['avg_reward'])
		a_c_co.append(co_metrics['avg_cost'])
		a_s_co.append(co_metrics['success_rate'])

		a_r_d=peril_metrics['avg_reward']-a_metrics['avg_reward']
		a_c_d=a_metrics['avg_cost']-peril_metrics['avg_cost']
		a_s_d=peril_metrics['success_rate']-a_metrics['success_rate']

		print("The reward gap is:",a_r_d)
		print("The cost gap is:",a_c_d)
		print("The success gap is:",a_s_d)

	rewards=[a_r,a_r_gt,a_r_cr,a_r_co]
	costs=[a_c,a_c_gt,a_c_cr,a_c_co]
	success_rates=[a_s,a_s_gt,a_s_cr,a_s_co]

	dtio.write_csv("test.csv",rewards+costs+success_rates)
	data=dtio.read_csv("test.csv")
	means=[]
	errs=[]
	for row in data:
		  row=list(row)
		  print(row)
		  means.append(sum(row)/len(row))
		  errs.append(st.stdev(row))
	"""
	mean_a_r=sum(a_r)/len(a_r)
	mean_a_c=sum(a_c)/len(a_c)
	mean_a_s=sum(a_s)/len(a_s)

	mean_a_r_gt=sum(a_r_gt)/len(a_r_gt)
	mean_a_c_gt=sum(a_c_gt)/len(a_c_gt)
	mean_a_s_gt=sum(a_s_gt)/len(a_s_gt)

	mean_a_r_cr=sum(a_r_cr)/len(a_r_cr)
	mean_a_c_cr=sum(a_c_cr)/len(a_c_cr)
	mean_a_s_cr=sum(a_s_cr)/len(a_s_cr)

	mean_a_r_co=sum(a_r_co)/len(a_r_co)
	mean_a_c_co=sum(a_c_co)/len(a_c_co)
	mean_a_s_co=sum(a_s_co)/len(a_s_co)


	r=[mean_a_r,mean_a_r_gt,mean_a_r_cr,mean_a_r_co]
	c=[mean_a_c,mean_a_c_gt,mean_a_c_cr,mean_a_c_co]
	s=[mean_a_s,mean_a_s_gt,mean_a_s_cr,mean_a_s_co]

	err_r=[st.stdev(a_r),st.stdev(a_r_gt),st.stdev(a_r_cr),st.stdev(a_r_co)]
	err_c=[st.stdev(a_c),st.stdev(a_c_gt),st.stdev(a_c_cr),st.stdev(a_c_co)]
	err_s=[st.stdev(a_s),st.stdev(a_s_gt),st.stdev(a_s_cr),st.stdev(a_s_co)]
	"""
	r=means[:4]
	c=means[4:8]
	s=means[8:]
	err_r=errs[:4]
	err_c=errs[4:8]
	err_s=errs[8:]

	baselines=["Act on\n uniform distribution","Accurate PERIL","Accurate Perception\n Only","Accurate Reasoner\n Only"]
	plt.figure()
	#plt.subplots()
	plt.rcParams.update({'font.size': 8})
	#plt.rcParams.update({'labelsize': 8})
	plt.grid(True,alpha=0.1)
	bars=plt.bar(baselines,r,yerr=err_r)
	for bar in bars:
		height=bar.get_height()
		plt.text(bar.get_x()+bar.get_width()/2., 1.05*height, "%d" % int(height), ha="center", va="bottom")
	plt.title('Rewards by acting on different belief')
	plt.ylabel("Reward")
	#plt.ylim(20,)
	plt.savefig("pomdp_r.pdf")
	plt.close()

	plt.figure()
	#plt.subplots()
	bars=plt.bar(baselines,c,yerr=err_c)
	for bar in bars:
		height=bar.get_height()
		plt.text(bar.get_x()+bar.get_width()/2., 1.05*height, "%d" % int(height), ha="center", va="bottom")
	plt.ylabel("Cost")

	#plt.ylim(5,)
	plt.savefig("pomdp_c.pdf")

	plt.figure()
	#plt.subplots()
	bars=plt.bar(baselines,s,yerr=err_s)
	for bar in bars:
		height=bar.get_height()
		plt.text(bar.get_x()+bar.get_width()/2., 1.02*height, round(height,2), ha="center", va="bottom")
	plt.ylabel("Successrate")
	plt.grid(True,alpha=0.1)
	#plt.ylim(0.7,)
	plt.savefig("pomdp_s.pdf")
	plt.close()

def main():
	planner=Planner()
	sim=AbstractSimulator()
	test_data=sim.minor_generate_data(5000)
	minor_a_test_err(sim,test_data)
	#sim.minor_a(planner,test_data)
	#sim.minor_a_gt(planner,test_data)
	#sim.minor_a_cr(planner,test_data)
	#sim.minor_a_co(planner,test_data)



if __name__ == '__main__':
	main()
