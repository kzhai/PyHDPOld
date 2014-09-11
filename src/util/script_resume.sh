#!/bin/bash

#DATA_PATH="140730-224558-k10-d1000-vpk10-wpd30-False"
DATA_PATH="140911-110426-k5-d200-v20-wpd30-False"
#DATA_PATH="140911-141458-k5-d500-v20-wpd30-False"

SNAPSHOT_INDEX=1000
TRAINING_ITERATION=2000
SUBSTRING_INDEX=62

PROJECT_HOME="/windows/d/Workspace/PyHDP"

cd $PROJECT_HOME\/src
#mkdir $PROJECT_HOME\/output\/$DATA_PATH

for MODEL_DIRECTORY in $PROJECT_HOME\/output\/$DATA_PATH\/*
do
	if [ -d "$MODEL_DIRECTORY" ]
	then
		#echo "Processing $MODEL_DIRECTORY"
		MODEL_SETTINGS=`basename $MODEL_DIRECTORY`
		#echo "$MODEL_SETTINGS"
		INFERENCE_MODE=${MODEL_SETTINGS:$SUBSTRING_INDEX}
		echo "$INFERENCE_MODE"
		
		nohup python -O \
			-um hdp.resume \
    		--input_directory=$MODEL_DIRECTORY \
			--snapshot_index=$SNAPSHOT_INDEX \
			--training_iteration=$TRAINING_ITERATION \
			> $PROJECT_HOME\/output/$DATA_PATH\/nohup.$INFERENCE_MODE\.$TRAINING_ITERATION\.out &

		sleep 1

	fi
done