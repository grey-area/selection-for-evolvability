#!/usr/bin/env python

import os
import itertools
import random

data_directory = '../processed_data/1_split_by_ff_and_method/'
dest_directory = '../processed_data/' +  os.path.basename(__file__).split('.')[0] + '/'

to_remove = os.listdir(dest_directory)
for filename in to_remove:
    os.remove(dest_directory + filename)

fitness_functions = ['simple','mask','symmetry','lipson']
methods = ['point','kalman','particle']
pairs = list(itertools.product(fitness_functions, methods))

triples2 = list(itertools.product(['f','e'],['FvsE','FvsPTS','EvsPTS'],['train','test']))

file_dict = {}

for fitness_function, method in pairs:

    if method == 'point':
        pairs2 = list(itertools.product(['f','e'],['FvsE']))
    else:
        pairs2 = list(itertools.product(['f','e'],['FvsE','FvsPTS','EvsPTS']))

    for recording1, recording2 in pairs2:
        for t in ['train','test']:
            file_dict[(fitness_function, method, recording1, recording2, t)] = open('%s%s_%s_%s_%s_%s.dat' % (dest_directory, fitness_function, method, recording1, recording2, t),'w')
    
    with open('%s%s_%s.dat' % (data_directory, fitness_function, method)) as f:
        header = f.readline()

        header_fields = header.split(',')

        if method == 'point':
            header = ','.join(header_fields[2:8]) + ',Termination heuristic on,Termination heuristic type,' + ','.join(header_fields[9:12]) + ',Result,Result sign\n'
        else:
            header = ','.join(header_fields[2:8]) + ',Termination heuristic on,Termination heuristic type,' + ','.join(header_fields[9:10]) + ',' + header_fields[12] + ',Result,Result sign\n'

        for r1, r2 in pairs2:
            for t in ['train','test']:
                file_dict[(fitness_function, method, r1, r2, t)].write(header)

        for line in f.readlines():
            fields = line.split(',')

            constructed_line = ','.join(fields[2:7]) + ',%d' % int(fields[7]=='maximum')
            constructed_line += ',%d,%d,' % (int(fields[8] != 'none'), int(fields[8] == 'relative'))
            if method == 'point':
                constructed_line += ','.join(fields[9:12]) + ','
            else:
                constructed_line += ','.join(fields[9:10]) + ',' + fields[12] + ','

            for r1, r2 in pairs2:
                index1 = -6
                index2 = -4

                if r2 == 'FvsPTS':
                    index2 += 2
                elif r2 == 'EvsPTS':
                    index1 += 2
                    index2 += 2
                if r1 == 'e':
                    index1 += 1
                    index2 += 1
                    
                value = float(fields[index2]) - float(fields[index1])
                this_constructed_line = constructed_line + '%f,%d\n' % (value, int(value>0))
                
                if random.random() < 0.5:
                    t = 'train'
                else:
                    t = 'test'

                file_dict[(fitness_function, method, r1, r2, t)].write(this_constructed_line)
            
for f in file_dict.values():
    f.close()
