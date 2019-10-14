"""
Pricing Analysis: create plots for compute allocation for user workloads
Compute options available: Reserved instances (varying durations), Serverless Computing
VM : M/G/1 model
SC: M/G/infinity model
Plots: VM only, SC only, VM + SC

Usage: python analysis.py <workload>
<workload>: w1,w2,w3,w4,w5,w6,w7,wAll

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

markers = ['s', 'h', '^', '*', 'o', 'p', '+', 'x', '<', 'D', '>', 'v', 'd', 0, 5, 2, 7, 1, 4, 3, 6, '1', '2', '3', '4', '8']


# Greedy method to calculate the number of VMs
def vm_only(totalload, cfg):
	# TODO
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
		provision_vms = math.ceil(minLoad/mu_v)
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
	num_vms = [] # number of VMs initiated at each time stamp
	sc_load = [] # load on SC at each time stamp
	vm_cost = 0
	sc_cost = 0
	return num_vms, sc_load, vm_cost, sc_cost

def get_workload(workload_type, cfg):
	workload = [] # load for each timestamp
	if workload_type == 'w1':	# Facebook Hadoop
		workload = []
	elif workload_type == 'test':	# Facebook Hadoop
		workload = [1,2,0,1]
	elif workload_type == 'zipf':	# Zipf distribution, parameters specified in config.yml
		multiple_workloads = [np.random.zipf(zipf_alpha, cfg['zipf']['samples_to_generate']) for zipf_alpha in cfg['zipf']['zipf_alphas']]
		workload = multiple_workloads[2]
		workload = workload[workload<cfg['zipf']['max_lambda']][:cfg['zipf']['samples_to_use']]
		# print len(workload)
		# count, bins, ignored = plt.hist(workload[workload<50], 50, normed=True)
		# x = np.arange(1., 50.)
		# y = x**(-zipf_alpha) / special.zetac(zipf_alpha)
		# plt.plot(x, y/max(y), linewidth=2, color='r')
		# plt.show()
	elif workload_type == 'uniform':	# uniform distribution, parameters specified in config.yml
		workload = np.random.random_integers(cfg['uniform']['min_lambda'],cfg['uniform']['max_lambda'],cfg['uniform']['samples_to_use'])
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

	totalload = get_workload(workload_type, cfg)
	# print totalload
	num_vms, vm_cost = vm_only(totalload, cfg)
	# print totalload
	sc_load, sc_cost = sc_only(totalload, cfg)
	# print totalload
	num_vms_hybrid, sc_load_hybrid, vm_cost_hybrid, sc_cost_hybrid = vm_sc(totalload, cfg)
	# print totalload
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