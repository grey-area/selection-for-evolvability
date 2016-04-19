#!/usr/bin/env python

import sys, os

trials = 1
if len(sys.argv) > 1:
    trials = int(sys.argv[1])
trials_per = 1
if len(sys.argv) > 2:
    trials_per = int(sys.argv[2])
fitness_function_name = "simple"
if len(sys.argv) > 3:
    fitness_function_name = sys.argv[3]

with open("clstr/jobs.txt", "w") as f:
    for trial in xrange(trials):
        exec_string = "cd %s; " % os.getcwd()
        exec_string += "julia experiment.jl %d %d %s\n" % (trial, trials_per, fitness_function_name)
        f.write(exec_string)
