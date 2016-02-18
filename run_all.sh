#!/bin/bash

python gen_jobs.py
./run_clstr.sh
python collate.py
./cleanup.sh
