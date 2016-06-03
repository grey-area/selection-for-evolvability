#!/bin/bash

./cleanup.sh
cat clstr/jobs.txt | bash
wait
