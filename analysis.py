"""
Pricing Analysis: create plots for compute allocation for user workloads
Compute options available: Reserved instances (varying durations), Serverless Computing
VM : M/G/1 model
SC: M/G/infinity model
Plots: VM only, SC only, VM + SC

Usage: python analysis.py <config.yml> <workload>
<workload>: w1,w2,w3,w4,w5,w6,w7,wAll,zipf, uniform

Author: Kunal Mahajan
PhD student
mkunal@cs.columbia.edu
Columbia University
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy
import math
import yaml
import numpy as np
from scipy import special
import random

sys.setrecursionlimit(1500)

markers = ['s', 'h', '^', '*', 'o', 'p', '+', 'x', '<', 'D', '>', 'v', 'd', 0, 5, 2, 7, 1, 4, 3, 6, '1', '2', '3', '4', '8']

def plot_zipf_workload(workload, zipf_alpha):
	print len(workload)
	count, bins, ignored = plt.hist(workload[workload<50], 50, normed=True)
	x = np.arange(1., 50.)
	y = x**(-zipf_alpha) / special.zetac(zipf_alpha)
	plt.plot(x, y/max(y), linewidth=2, color='r')
	plt.show()

# Greedy method to calculate the number of VMs
def vm_only(totalload, cfg):
	workload = [load for load in totalload]
	num_vms = [0 for load in workload] # number of VMs initiated at each time stamp
	vm_cost = 0
	window = cfg['vm']['window']
	mu_v = cfg['vm']['mu_v']
	alpha_v = cfg['vm']['alpha_v']
	
	while sum(workload) != 0:
		window_sums = [(sum(workload[i:i+window]),i) for i in range(len(workload)-window+1)]
		index = max(window_sums,key=lambda item:item[0])[1]
		minLoad = float("inf")
		for i in range(index, index+window):
			minLoad = min(minLoad, workload[i]) if workload[i] != 0 else minLoad
		provision_vms = math.ceil(float(minLoad)/mu_v)
		num_vms[index] += provision_vms
		for i in range(index, index+window):
			workload[i] = max(0,workload[i]-(provision_vms*mu_v))
	vm_cost = sum(num_vms) * alpha_v
	return num_vms, vm_cost

def sc_only(totalload, cfg):
	workload = [load for load in totalload]
	sc_load = workload # load on SC at each time stamp
	sc_cost = cfg['sc']['alpha_s'] * (sum(workload) / cfg['sc']['mu_s'])
	return sc_load, sc_cost



def vm_sc(totalload, cfg):
	# TODO
	workload = [load for load in totalload]
	num_vms = [0 for load in workload] # number of VMs initiated at each time stamp
	sc_load = [0 for load in workload] # load on SC at each time stamp
	vm_cost = 0
	sc_cost = 0
	window = cfg['vm']['window']
	mu_v = cfg['vm']['mu_v']
	alpha_v = cfg['vm']['alpha_v']
	alpha_s = cfg['sc']['alpha_s']
	mu_s = cfg['sc']['mu_s']

	# Apply the Lemma first
	i = 0
	while i < len(workload) - window + 1:
		if all(workload[j] >= mu_v for j in range(i, i+window)):
			num_vms[i] += 1
			for j in range(i, i+window):
				workload[j] -= mu_v
		else:
			i += 1
	# Apply DP
	memo = {} 

	def serialize(load):
		return "_".join(str(l) for l in load)		

	def deserialize(load_string):
		return load_string.split("_")

	def dp(t, load):
		# print memo
		load_string = serialize(load)
		if (t,load_string) in memo:
			return memo[(t,load_string)]

		vm_possible = int(math.ceil(float(load[t])/mu_v))
		total_cost = []
		if t == len(load) - 1:
			for num_vm in range(0, vm_possible + 1):
				total_cost.append((alpha_v * num_vm) + (alpha_s * max(0, load[t] - (num_vm * mu_v)) / float(mu_s)))
		else:
			for num_vm in range(0, vm_possible + 1):
				cost_t = (alpha_v * num_vm) + (alpha_s * max(0, load[t] - (num_vm * mu_v)) / float(mu_s))
				new_load = [l for l in load]
				# print len(new_load), len(load)
				for i in range(t, t+window):
					if i >= len(new_load):
						break
					new_load[i] = max(0, load[i] - (num_vm * mu_v))
				cost_t_plus_1, _ = dp(t+1, new_load)
				total_cost.append(cost_t + cost_t_plus_1)

		memo[(t,load_string)] = min(total_cost)
		return min(total_cost), total_cost.index(min(total_cost))
			# 	memo[(t, num_vm, load)] = (alpha_v * num_vm) + (alpha_s * max(0, load[t] - (num_vm * mu_v)) / mu_s)
			# min_cost, optimal_num_vm =  min([(memo[(t,i,load)],i) for i in range(0, vm_possible+1)])
			# return min_cost, optimal_num_vm
	
	min_cost, _ = dp(0, workload)
	# add number of VMs from Lemma
	min_cost += alpha_v * sum(num_vms)
	return num_vms, sc_load, min_cost, sc_cost # DO NOT USE THIS
	# return num_vms, sc_load, vm_cost, sc_cost

#################################################    WORKLOAD GENERATOR     #################################################

"""
Generate job arrival times specified by Poisson process with a given lambda
If isTrace == True, merge job arrivals. Else generate job arrivals and then merge
"""
def gen_arrivals(isTrace, val_lambda, num_arrivals, num_intervals):
	
	arrival_times = []
	arrivals = [0 for i in range(num_intervals)]
	time = 0

	# Generate arrival times
	if not isTrace:
		for i in range(num_arrivals):
			p = random.random()			#Get the next probability value from Uniform(0,1)
			inter_arrival_time = -math.log(1.0 - p)/val_lambda		#Plug it into the inverse of the CDF of Exponential(_lambda)
			time += inter_arrival_time		#Add the inter-arrival time to the running sum
			arrival_times.append(time)
	else:
		# TODO TRACE DRIVEN
		arrival_times = []

	# Merge the arrivals into intervals
	interval_length = num_arrivals/num_intervals
	print interval_length
	error = 0
	for time in arrival_times:
		if int(time/interval_length) >= len(arrivals):
			error += 1
			continue
		arrivals[int(time/interval_length)] += 1
	print "jobs generated but not used = %d" % error
	return arrivals

"""
compute the actual demand for each time interval
"""
def gen_load_per_time(arrivals, job_sizes):
	if len(job_sizes) < sum(arrivals):
			print "Error: number of job_sizes is less than number of jobs"
			return
	load = [0 for val in arrivals]
	curr = 0
	for i in range(len(arrivals)):
		for size in job_sizes[curr:curr+arrivals[i]]:
			j = i
			while j < i+size:
				# if j >= len(load):
				# 	load.append(1)
				# else:
				# 	load[j] += 1		# Each job uses capacity of 1 in each time interval
				if j < len(load):
					load[j] += 1		# Each job uses capacity of 1 in each time interval
				j += 1
		curr += arrivals[i]
	return load

"""
Workload Generator based on workload type
For simulated workloads, job arrival times specified by Poisson process
For trace-driven workloads, job arrival times are from the trace
"""
def get_workload(workload_type, cfg):
	# TODO
	workload = [] # load for each timestamp
	if workload_type == 'w1':	# Facebook Hadoop
		workload = []
	elif workload_type == 'test':	# Facebook Hadoop
		workload = [1,2,0,1]
	elif workload_type == 'zipf':	# Zipf distribution, parameters specified in config.yml

		arrivals = gen_arrivals(False, cfg['zipf']['val_lambda'], cfg['zipf']['jobs_to_generate'], cfg['zipf']['num_intervals'])
		multiple_workloads = [np.random.zipf(zipf_alpha, cfg['zipf']['job_sizes_to_generate']) for zipf_alpha in cfg['zipf']['zipf_alphas']]
		job_sizes = multiple_workloads[2]
		job_sizes = job_sizes[job_sizes<cfg['zipf']['max_job_size']][:cfg['zipf']['jobs_to_generate']]
		workload = gen_load_per_time(arrivals, job_sizes)
		print len(arrivals), len(job_sizes), len(workload)
	elif workload_type == 'uniform':	# uniform distribution, parameters specified in config.yml
		job_sizes = np.random.random_integers(cfg['uniform']['min_job_size'],cfg['uniform']['max_job_size'],cfg['uniform']['jobs_to_generate'])
		# workload = [3,4,1,3,3] # Give different result for VM only and VM+SC
	return workload

def main():
	if len(sys.argv) <= 2:
		print "USAGE: python analysis.py <config.yml> <workload>"
		print "<workload>: w1,w2,w3,w4,w5,w6,w7,wAll,zipf,uniform"
		return
	config_file = sys.argv[1]
	workload_type = sys.argv[2]
	with open(config_file, 'r') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

	print "%s workload: generating" % workload_type
	totalload = get_workload(workload_type, cfg)
	print "%s workload: generated" % workload_type
	print totalload
	print "Executing VM only case"
	num_vms, vm_cost = vm_only(totalload, cfg)
	# print totalload
	print "Executing SC only case"
	sc_load, sc_cost = sc_only(totalload, cfg)
	# print totalload
	print "Executing VM+SC case"
	num_vms_hybrid, sc_load_hybrid, vm_cost_hybrid, sc_cost_hybrid = vm_sc(totalload, cfg)
	print "Plotting results"
	# TODO: plot the above data

	results = "********** COST (Workload: %s) *********  SC only: %f\t VM only: %f\t VM+SC: %f" % (workload_type, sc_cost, vm_cost, vm_cost_hybrid+sc_cost_hybrid)
	print results
	filename = './graphs/' + workload_type + '.png'
	fig = plt.figure()
	# legends = []
	# for cp_ratio in cp_ratios:
	# 	key = r'$\alpha_{s\_cp}$=' + str(cp_ratio) + r'$\alpha_{v\_cp}$'
	# 	legends.append(key)
	plt.subplot(2,1,1)
	# print num_vms
	# print [i for i in range(len(totalload))]
	plt.plot([i for i in range(len(totalload))], num_vms, 'c*', markersize=7)
	# plt.plot(price_ratios[::20], , 'ro', markersize=7)
	# plt.plot(price_ratios[::20], results[2][::20], 'g^', markersize=7)
	# plt.plot(price_ratios[::200], results[3][::200], 'bs', markersize=7)
	# plt.plot(price_ratios, results[0], 'c', linewidth='2')
	# plt.plot(price_ratios, results[1], 'r', linewidth='2')
	# plt.plot(price_ratios, results[2], 'g', linewidth='2')
	# plt.plot(price_ratios, results[3], 'b', linewidth='2')

	# plt.legend(legends, loc='upper right', fontsize=21)
	plt.ylabel('Number of VMs', fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	plt.subplot(2,1,2)
	plt.plot([i for i in range(len(totalload))], sc_load, 'ro', markersize=7)
	plt.ylabel('SC Load', fontsize=25)


	plt.xlabel('Time', fontsize=25)
	plt.savefig(filename)

if __name__ == '__main__':
	main()