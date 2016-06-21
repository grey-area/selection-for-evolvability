#!/usr/bin/env python

# samples, trials, append

import sys, os

samples = 1
if len(sys.argv) > 1:
    samples = int(sys.argv[1])
trials = 1
if len(sys.argv) > 2:
    trials = int(sys.argv[2])
append = False
if len(sys.argv) > 3:
    append = True

with open("clstr/jobs.txt", "w") as f:
    base_job_id = 0
    if append:
        files = os.listdir("data")
        if len(files) > 0:
            highest_job_id = max([int(filename.split('.')[0]) for filename in files])
            base_job_id = highest_job_id + 1

    for sample in xrange(samples):
        job_id = sample + base_job_id
        exec_string = "cd %s; " % os.getcwd()
        exec_string += "julia experiment.jl %d %d\n" % (job_id, trials)
        f.write(exec_string)
