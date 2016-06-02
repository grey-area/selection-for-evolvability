#!/usr/bin/env python

import sys

second = './collated_data/'

if len(sys.argv) < 2:
    print "Specify at least one collated data file."
    sys.exit(0)
else:
    first = sys.argv[1]

if len(sys.argv) > 2:
    second = sys.argv[2]

with open(first + "collated_data.dat") as f:
    first_text = f.read()

with open(second + "collated_data.dat") as f:
    second_text = f.read()

def read_text(text):
    entries = text.split("\n\n")
    results_dict = {}
    
    for entry in entries:
        if entry == "":
            continue
        
        title, data = entry.split("\n")
        mean_data = sum(float(i) for i in data.split(",")) / len(data)
        results_dict[title] = mean_data

    return results_dict

first_dict = read_text(first_text)
second_dict = read_text(second_text)

difference_dict = {}

for key in first_dict.keys():
    difference_dict[key] = abs(first_dict[key] - second_dict[key]) / second_dict[key]

ordered_differences = sorted(difference_dict.items(), key=lambda x: x[1], reverse=True)
ordered_differences = filter(lambda x: x[1] > 0.2, ordered_differences)
differences_only = [item[1] for item in ordered_differences]

field_dicts = {}

for key, value in ordered_differences:
    fields = key.split('+')
    for field in [2,4,7,8]:
        parts = fields[field].split("-")
        if parts[0] not in field_dicts.keys():
            field_dicts[parts[0]] = {}
        if parts[1] not in field_dicts[parts[0]].keys():
            field_dicts[parts[0]][parts[1]] = 0
        field_dicts[parts[0]][parts[1]] += 1

for key, value in field_dicts.items():
    print key
    for key, value2 in value.items():
        print key, value2
    print ""
    
#print len(differences_only)

#for key, value in ordered_differences:
#    print key, value

#import matplotlib.pyplot as plt
#plt.hist(differences_only, bins=80)
#plt.show()

