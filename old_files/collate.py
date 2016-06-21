#!/usr/bin/env python

import os
import sys

data = {}

for filename in os.listdir('data'):
    with open('data/%s' % filename ) as f:
        text = f.read()

    entries = text.split("\n\n")

    for entry in entries:

        if entry == "":
            continue

        lines = entry.split("\n")
        key = lines[0].replace('\n','')
        value = lines[1].replace('\n','')
        if key in data.keys():
            data[key].append(value)
        else:
            data[key] = [value]

with open('collated_data/collated_data.dat', 'w') as f:
    for k,v in data.items():
        f.write(k)
        f.write('\n')
        f.write(','.join(v))
        f.write('\n\n')

