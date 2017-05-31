#!/usr/bin/env python

import os
import itertools

data_directory = '../data/'
dest_directory = '../processed_data/' +  os.path.basename(__file__).split('.')[0] + '/'

to_remove = os.listdir(dest_directory)
for filename in to_remove:
    os.remove(dest_directory + filename)

fitness_functions = ['simple','mask','symmetry','lipson']
methods = ['point','kalman','particle']
pairs = list(itertools.product(fitness_functions, methods))

data_dictionary = dict()
for pair in pairs:
    data_dictionary[pair] = ''
heading = None

filenames = os.listdir('../data/')
for f_i, filename in enumerate(filenames):
    with open(data_directory + filename) as f:
        first_line = f.readline()
        if heading is None:
            heading = first_line
        for line in f.readlines():
            fields = line.split(',')
            for fitness_function, method in pairs:
                if fields[0] == fitness_function and fields[1] == method:
                    data_dictionary[(fitness_function, method)] += line

for fitness_function, method in pairs:
    with open('%s%s_%s.dat' % (dest_directory, fitness_function, method), 'w') as f:
        f.write(heading)
        f.write(data_dictionary[(fitness_function, method)])
