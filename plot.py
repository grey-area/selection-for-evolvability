#!/usr/bin/env python

import numpy as np
import sys

with open('collated_data/collated_data.dat') as f:
    text = f.read()

blocks = text.split('\n\n')

params_list = []
data_list = []

file_strings = []

for block in blocks:
    if block == "":
        continue
    parts = block.split('\n')
    param_string = parts[0]
    file_string = '+'.join(param_string.split('+')[:2])
    if file_string not in file_strings:
        file_strings.append(file_string)
    data = np.array([float(d) for d in parts[1].split(',')])

    params_list.append(param_string)
    data_list.append(data)

means = np.array([d.mean() for d in data_list])
order = means.argsort()[::-1]

import matplotlib.pyplot as plt

ordered_params_lists = {}
ordered_data_lists = {}

for index in order:
    key = '+'.join(params_list[index].split('+')[:3])
    if key not in ordered_params_lists.keys():
        ordered_params_lists[key] = []
        ordered_data_lists[key] = []

    ordered_params_lists[key].append( '\n'.join(params_list[index].split('+')[3:]).replace('selection_type','').replace('heuristic','h').replace('evolvability_type','e').replace('problem','prob').replace('simple','s') )
    ordered_data_lists[key].append(data_list[index])

#colours = ['lightgreen','tan','pink',']
colours = {'fitness': 'lightgreen', 'point': 'tan', 'kalman': 'pink', 'particle': 'red'}

for key in ordered_params_lists.keys():
    num_to_show = 22

    box = plt.boxplot(ordered_data_lists[key][:num_to_show], notch=True, patch_artist=True)

    plt.title(key)
    plt.xticks(np.arange(num_to_show)+1, ordered_params_lists[key][:num_to_show])
    these_colours = []
    for patch, param in zip(box['boxes'], ordered_params_lists[key][:num_to_show]):
        c = colours[param.split('\n')[1].split('-')[1]]
        patch.set_facecolor(c)


    fig = plt.figure(1, figsize=(15,6))
    fig.savefig('plots/%s.png' % key, bbox_inches='tight')
    plt.close()

