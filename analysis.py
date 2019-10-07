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


markers = ['s', 'h', '^', '*', 'o', 'p', '+', 'x', '<', 'D', '>', 'v', 'd', 0, 5, 2, 7, 1, 4, 3, 6, '1', '2', '3', '4', '8']



def vm_only(workload):
	# TODO
	num_vms = [] # number of VMs initiated at each time stamp
	vm_cost = 0
	return num_vms, vm_cost

def sc_only(workload):
	sc_load = workload # load on SC at each time stamp
	# TODO
	sc_cost = 0
	return sc_load, sc_cost

def vm_sc(workload):
	# TODO
	num_vms = [] # number of VMs initiated at each time stamp
	sc_load = [] # load on SC at each time stamp
	vm_cost = 0
	sc_cost = 0
	return num_vms, sc_load, vm_cost, sc_cost

def get_workload(workload_type):
	workload = [] # load for each timestamp
	return workload

def main():
	if len(sys.argv) <= 2:
		print "USAGE: python analysis.py <config.yml> <workload>"
		print "<workload>: w1,w2,w3,w4,w5,w6,w7,wAll"
		return
	config_file = sys.argv[1]
	workload_type = sys.argv[2]

	with open(config_file, 'r') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	print cfg['vm']['mu']

	num_vms, vm_cost = vm_only(workload_type)
	sc_load, sc_cost = sc_only(workload_type)
	num_vms, sc_load, vm_cost, sc_cost = vm_sc(workload_type)

	# TODO: plot the above data

if __name__ == '__main__':
	main()