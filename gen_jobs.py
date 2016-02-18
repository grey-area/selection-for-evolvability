#!/usr/bin/env python

import sys, os

trials = 50
if len(sys.argv) > 1:
    trials = int(sys.argv[1])
trials_per = 2
if len(sys.argv) > 2:
    trials_per = int(sys.argv[2])

with open("clstr/jobs.txt", "w") as f:
    for trial in xrange(trials):
        exec_string = "cd %s; " % os.getcwd()
        exec_string += "julia experiment.jl %d %d\n" % (trial, trials_per)
        f.write(exec_string)
