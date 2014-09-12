#!/bin/bash

#DATA_PATH="140730-224558-k10-d1000-vpk10-wpd30-False"
#DATA_PATH="140911-141458-k5-d500-v20-wpd30-False"
DATA_PATH="140912-100218-k5-d200-v20-wpd30-False"
TRAINING_ITERATION=2000
ALPHA_ALPHA=0.1

PROJECT_HOME="/windows/d/Workspace/PyHDP"

cd $PROJECT_HOME\/src
mkdir $PROJECT_HOME\/output\/$DATA_PATH

nohup python -O \
    -um hdp.launch \
    --input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
    --output_directory=$PROJECT_HOME\/output/ \
    --alpha_alpha=$ALPHA_ALPHA \
    --training_iteration=$TRAINING_ITERATION \
    > $PROJECT_HOME\/output/$DATA_PATH\/nohup.out &
	
sleep 1

nohup python -O \
    -um hdp.launch \
    --input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
    --output_directory=$PROJECT_HOME\/output/ \
    --training_iteration=$TRAINING_ITERATION \
    --alpha_alpha=$ALPHA_ALPHA \
    --split_merge_heuristics=0 \
    > $PROJECT_HOME\/output/$DATA_PATH\/nohup.smh0.out &
	
sleep 1

nohup python -O \
    -um hdp.launch \
    --input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
    --output_directory=$PROJECT_HOME\/output/ \
    --training_iteration=$TRAINING_ITERATION \
    --alpha_alpha=$ALPHA_ALPHA \
    --split_merge_heuristics=1 \
    > $PROJECT_HOME\/output/$DATA_PATH\/nohup.smh1-sp0-mp0.out &
	
sleep 1

nohup python -O \
    -um hdp.launch \
    --input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
    --output_directory=$PROJECT_HOME\/output/ \
    --training_iteration=$TRAINING_ITERATION \
    --alpha_alpha=$ALPHA_ALPHA \
    --split_merge_heuristics=1 \
    --split_proposal=1 \
    --merge_proposal=1 \
    > $PROJECT_HOME\/output/$DATA_PATH\/nohup.smh1-sp1-mp1.out &
	
sleep 1

nohup python -O \
    -um hdp.launch \
    --input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
    --output_directory=$PROJECT_HOME\/output/ \
    --training_iteration=$TRAINING_ITERATION \
    --alpha_alpha=$ALPHA_ALPHA \
    --split_merge_heuristics=1 \
    --split_proposal=2 \
    --merge_proposal=0 \
    > $PROJECT_HOME\/output/$DATA_PATH\/nohup.smh1-sp2-mp0.out &