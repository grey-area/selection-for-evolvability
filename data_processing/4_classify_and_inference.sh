#!/bin/bash

rm ../processed_data/4_classify_and_inference/*

python classify_and_inference.py 0 &
python classify_and_inference.py 1 &
python classify_and_inference.py 2 &
python classify_and_inference.py 3 &
wait

cp ../processed_data/4_classify_and_inference/* /home/andrew/phd/thesis/data/pickled/
