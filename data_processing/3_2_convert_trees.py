#!/usr/bin/env python

import subprocess

data_directory = '../processed_data/3_decision_trees/'
dest_directory = '/home/andrew/phd/thesis/data/trees/'

for fitness_function in ['simple','mask','symmetry','lipson']:
    for method in ['point','kalman','particle']:
        for r1 in ['f','e']:
            if method == 'point':
                r2s = ['FvsE']
            else:
                r2s = ['FvsE','FvsPTS','EvsPTS']
            for r2 in r2s:
                subprocess.call(["dot","-Tpdf","%s%s_%s_%s_%s.dot" % (data_directory, fitness_function, method, r1, r2), "-o", "%s%s_%s_%s_%s.pdf" % (dest_directory, fitness_function, method, r1, r2)])

