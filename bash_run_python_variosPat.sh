#!/bin/bash

for i in {1..18} #! e.g. of use: 1, {1,5,9,11,24}, and {1..18}.
do
	python script_AZC_Classification.py $i &
done