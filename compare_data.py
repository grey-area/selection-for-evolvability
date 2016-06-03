#!/usr/bin/env python

# Compare the first collated data file with the rest

import sys

# 0 = histogram
# 1 = which categories have greater than 15% absolute difference?
display_type = 0

if len(sys.argv) < 3:
    print "Specify at least two collated data files."
    sys.exit(0)
else:
    first = sys.argv[1]

rest = sys.argv[2:]

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

with open(first + "collated_data/collated_data.dat") as f:
    first_text = f.read()
first_dict = read_text(first_text)

differences_list = []
ordered_lists = []
for other in rest:
    with open(other + "collated_data/collated_data.dat") as f:
        other_text = f.read()
    other_dict = read_text(other_text)

    difference_dict = {}
    for key in first_dict.keys():
        difference_dict[key] = abs(first_dict[key] - other_dict[key]) / first_dict[key]
    ordered_differences = sorted(difference_dict.items(), key=lambda x: x[1], reverse=True)
    if display_type == 1:
        ordered_differences = filter(lambda x: x[1] > 0.15, ordered_differences)
    differences_only = [item[1] for item in ordered_differences]
    ordered_lists.append(ordered_differences)
    differences_list.append(differences_only)

if display_type == 1:
    for ordered_differences in ordered_lists:
        field_dicts = {}
        for key, value in ordered_differences:
            fields = key.split('+')
            for field in range(12):
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
else:
    import matplotlib.pyplot as plt
    for differences_only in differences_list:
        print len(differences_only)
        plt.hist(differences_only, bins=80, normed=True)
    plt.show()
