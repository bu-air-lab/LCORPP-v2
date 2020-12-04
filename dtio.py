import csv
import json
import matplotlib.pyplot as plt
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import os

def write_csv(file_name,data):
	with open(file_name,mode="w",newline='') as f:
		f_writer=csv.writer(f)
		f_writer.writerows(data)

def read_csv(file_name):
	data=[]
	with open(file_name) as f:
		f_reader=csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
		for row in f_reader:
			data.append(row)
	return data

def write_row(file_name,row):
	with open(file_name,mode="a",newline='') as f:
		f_writer=csv.writer(f)
		f_writer.writerow(row)

def get_means_err(file_name):
	mat=np.array(read_csv(file_name))
	means=np.mean(mat,axis=0)
	err=np.std(mat,axis=0)
	return means,err

def plot_learning(datasizes):
	peril_mean_r,peril_err_r=get_means_err("peril_rewards.csv")
	peril_mean_c,peril_err_c=get_means_err("peril_costs.csv")
	peril_mean_s,peril_err_s=get_means_err("peril_successrates.csv")

	plearn_mean_r,plearn_err_r=get_means_err("plearn_rewards.csv")
	plearn_mean_c,plearn_err_c=get_means_err("plearn_costs.csv")
	plearn_mean_s,plearn_err_s=get_means_err("plearn_successrates.csv")

	rlearn_mean_r,rlearn_err_r=get_means_err("rlearn_rewards.csv")
	rlearn_mean_c,rlearn_err_c=get_means_err("rlearn_costs.csv")
	rlearn_mean_s,rlearn_err_s=get_means_err("rlearn_successrates.csv")

	plt.figure()
	plt.errorbar(datasizes,peril_mean_r,yerr=peril_err_r,capsize=2,label="PREIL Learning")
	plt.errorbar(datasizes,rlearn_mean_r,yerr=rlearn_err_r,capsize=2,ls='--',label="Reasoner Learning only")
	plt.errorbar(datasizes,plearn_mean_r,yerr=plearn_err_r,capsize=2,ls='--',label="Perception Learning only")
	#plt.xscale('log',base=2)
	plt.ylabel("Reward")
	plt.xlabel("Size of Data")
	#plt.ylim(30,)
	plt.legend(fontsize = 'x-small')
	plt.savefig("reward_curve.pdf")
	plt.close()

	plt.figure()
	plt.errorbar(datasizes,peril_mean_c,yerr=peril_err_c,capsize=2,label="PREIL Learning")
	plt.errorbar(datasizes,rlearn_mean_c,yerr=rlearn_err_c,capsize=2,ls='--',label="Reasoner Learning only")
	plt.errorbar(datasizes,plearn_mean_c,yerr=plearn_err_c,capsize=2,ls='--',label="Perception Learning only")
	#plt.xscale('log',base=2)
	plt.xlabel("Size of Data")
	plt.ylabel("Cost")
	plt.legend(fontsize = 'x-small')
	#plt.ylim(50,80)
	plt.savefig("cost_curve.pdf")
	plt.close()

	plt.figure()
	#plt.subplots()
	plt.errorbar(datasizes,peril_mean_s,yerr=peril_err_s,capsize=2,label="LPRA Learning")
	plt.errorbar(datasizes,rlearn_mean_s,yerr=rlearn_err_s,capsize=2,label="Reasoner Learning only")
	plt.errorbar(datasizes,plearn_mean_s,yerr=plearn_err_s,capsize=2,label="Perception Learning only")
	#plt.xscale('log',base=2)
	plt.xlabel("Size of Data")
	plt.ylabel("Success rate")
	#plt.ylim(0,6,)
	plt.legend(fontsize = 'x-small')
	plt.savefig("success_curve.pdf")
	plt.close()

def plot_learning_errband(datasizes):
	
	#plot error_bands
	peril_mean_r,peril_err_r=get_means_err("peril_rewards.csv")
	peril_mean_c,peril_err_c=get_means_err("peril_costs.csv")
	peril_mean_s,peril_err_s=get_means_err("peril_successrates.csv")

	plearn_mean_r,plearn_err_r=get_means_err("plearn_rewards.csv")
	plearn_mean_c,plearn_err_c=get_means_err("plearn_costs.csv")
	plearn_mean_s,plearn_err_s=get_means_err("plearn_successrates.csv")

	rlearn_mean_r,rlearn_err_r=get_means_err("rlearn_rewards.csv")
	rlearn_mean_c,rlearn_err_c=get_means_err("rlearn_costs.csv")
	rlearn_mean_s,rlearn_err_s=get_means_err("rlearn_successrates.csv")

	y1_rewards=list(peril_mean_r+peril_err_r)
	y2_rewards=list(peril_mean_r-peril_err_r)
	y1_rlearn_rewards=list(rlearn_mean_r+rlearn_err_r)
	y2_rlearn_rewards=list(rlearn_mean_r-rlearn_err_r)
	y1_plearn_rewards=list(plearn_mean_r+plearn_err_r)
	y2_plearn_rewards=list(plearn_mean_r-plearn_err_r)

	plt.figure()
	plt.plot(datasizes,peril_mean_r,marker='^',label="LPRA Learning")
	plt.fill_between(datasizes,y1_rewards,y2_rewards,alpha=0.5)

	plt.plot(datasizes,rlearn_mean_r,marker='s',ls='--',label="Reasoner Learning only")
	plt.fill_between(datasizes,y1_rlearn_rewards,y2_rlearn_rewards,alpha=0.5)

	plt.plot(datasizes,plearn_mean_r,marker='o',ls='--',label="Perception Learning only")
	plt.fill_between(datasizes,y1_plearn_rewards,y2_plearn_rewards,alpha=0.5)

	plt.ylabel("Reward")
	plt.xlabel("Size of Data")
	#plt.xscale('log',basex=2)
	#plt.ylim(30,)
	plt.legend(loc= 'lower right', fontsize = 'x-small')
	plt.savefig("reward_curveband.pdf")
	plt.close()

	y1_costs=list(peril_mean_c+peril_err_c)
	y2_costs=list(peril_mean_c-peril_err_c)
	y1_rlearn_costs=list(rlearn_mean_c+rlearn_err_c)
	y2_rlearn_costs=list(rlearn_mean_c-rlearn_err_c)
	y1_plearn_costs=list(plearn_mean_c+plearn_err_c)
	y2_plearn_costs=list(plearn_mean_c-plearn_err_c)

	plt.figure()
	plt.plot(datasizes,peril_mean_c,marker='^',label="PERIL Learning")
	plt.fill_between(datasizes,y1_costs,y2_costs,alpha=0.5)

	plt.plot(datasizes,rlearn_mean_c,marker='s',ls='--',label="Reasoner Learning only")
	plt.fill_between(datasizes,y1_rlearn_costs,y2_rlearn_costs,alpha=0.5)

	plt.plot(datasizes,plearn_mean_c,marker='o',ls='--',label="Perception Learning only")
	plt.fill_between(datasizes,y1_plearn_costs,y2_plearn_costs,alpha=0.5)

	plt.ylabel("Cost")
	plt.xlabel("Size of Data")
	#plt.xscale('log',basex=2)
	#plt.ylim(30,)
	plt.legend(loc= 'lower right', fontsize = 'x-small')
	plt.savefig("cost_curveband.pdf")
	plt.close()

	y1_rates=list(peril_mean_s+peril_err_s)
	y2_rates=list(peril_mean_s-peril_err_s)
	y1_rlearn_rates=list(rlearn_mean_s+rlearn_err_s)
	y2_rlearn_rates=list(rlearn_mean_s-rlearn_err_s)
	y1_plearn_rates=list(plearn_mean_s+plearn_err_s)
	y2_plearn_rates=list(plearn_mean_s-plearn_err_s)

	plt.figure()
	plt.plot(datasizes,peril_mean_s,marker='^',label="PERIL Learning")
	plt.fill_between(datasizes,y1_rates,y2_rates,alpha=0.5)

	plt.plot(datasizes,rlearn_mean_s,marker='s',ls='--',label="Reasoner Learning only")
	plt.fill_between(datasizes,y1_rlearn_rates,y2_rlearn_rates,alpha=0.5)

	plt.plot(datasizes,plearn_mean_s,marker='o',ls='--',label="Perception Learning only")
	plt.fill_between(datasizes,y1_plearn_rates,y2_plearn_rates,alpha=0.5)

	plt.ylabel("Rates of success")
	plt.xlabel("Size of Data")
	#plt.xscale('log',basex=2)
	#plt.ylim(30,)
	plt.legend(loc= 'lower right', fontsize = 'x-small')
	plt.savefig("success_curveband.pdf")
	plt.close()

def plot_comparison():

	reward,error_r=get_means_err("reward_comps.csv")
	cost,error_c=get_means_err("cost_comps.csv")
	success,error_s=get_means_err("successrate_comps.csv")
	#peril_rewards=read_csv("reward_comps.csv")
	baselines=["PERIL","PR+A-","R+A","A"]

	print(reward,cost,success)
	#print(stats.ttest_ind(LPRA_r,RA_r))
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

def clear_results_data():
	if os.path.exists("rlearn_cr.csv"):
		os.remove("rlearn_cr.csv")
	if os.path.exists("rlearn_co.csv"):
		os.remove("rlearn_co.csv")
	if os.path.exists("conf_matrix.txt"):
		os.remove("conf_matrix.txt")

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


def main():
	plot_learning([30,130,430,1230])
	plot_learning_errband([30,130,430,1230])
	plot_comparison()

if __name__ == '__main__':
	main()
