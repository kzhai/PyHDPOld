#!/bin/bash

#DATA_PATH="140730-224558-k10-d1000-vpk10-wpd30-False"
DATA_PATH="140806-165121-k5-d500-vpk5-wpd30-False"
TRAINING_ITERATION=500

PROJECT_HOME="/Users/kezhai/Workspace/PyHDP"

cd $PROJECT_HOME\/src
mkdir $PROJECT_HOME\/output\/$DATA_PATH

nohup python \
	-um hdp.launch \
	--input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
	--output_directory=$PROJECT_HOME\/output/ \
	--training_iteration=$TRAINING_ITERATION \
	> $PROJECT_HOME\/output/$DATA_PATH\/nohup.out &
	
sleep 1

nohup python \
	-um hdp.launch \
	--input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
	--output_directory=$PROJECT_HOME\/output/ \
	--training_iteration=$TRAINING_ITERATION \
	--split_merge_heuristics=0 \
	> $PROJECT_HOME\/output/$DATA_PATH\/nohup.smh0.out &
	
sleep 1

nohup python \
	-um hdp.launch \
	--input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
	--output_directory=$PROJECT_HOME\/output/ \
	--training_iteration=$TRAINING_ITERATION \
	--split_merge_heuristics=1 \
	> $PROJECT_HOME\/output/$DATA_PATH\/nohup.smh1.sp0.mp0.out &
	
sleep 1

nohup python \
	-um hdp.launch \
	--input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
	--output_directory=$PROJECT_HOME\/output/ \
	--training_iteration=$TRAINING_ITERATION \
	--split_merge_heuristics=1 \
	--split_proposal=1 \
	--merge_proposal=1 \
	> $PROJECT_HOME\/output/$DATA_PATH\/nohup.smh1.sp1.mp1.out &
	
sleep 1

nohup python \
	-um hdp.launch \
	--input_directory=$PROJECT_HOME\/input/$DATA_PATH/ \
	--output_directory=$PROJECT_HOME\/output/ \
	--training_iteration=$TRAINING_ITERATION \
	--split_merge_heuristics=1 \
	--split_proposal=2 \
	--merge_proposal=0 \
	> $PROJECT_HOME\/output/$DATA_PATH\/nohup.smh1.sp2.mp0.out &