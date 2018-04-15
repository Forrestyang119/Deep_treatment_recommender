#!/bin/bash
for ran_seed in {5..5}
do
    for i in {1..18}
    do
        python3 activity_predictor_intubation.py $i $ran_seed
    done

done

