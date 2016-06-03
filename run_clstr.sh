#!/bin/bash

./cleanup.sh

# Run cluster
clstr --status --node-file ./clstr/cs_machines1.txt --job-file ./clstr/jobs.txt
