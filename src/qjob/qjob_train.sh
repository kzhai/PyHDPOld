#!/bin/bash
#PBS -N
#PBS -l mem=4gb,walltime=24:00:00
#PBS -q batch
#PBS -M zhaikedavy@gmail.com
#PBS -m ae
#PBS -k e
#PBS -V

PYTHON_COMMAND=/opt/local/stow/python-2.7.2/bin/python

cd /fs/clip-lsbi/Workspace/PyHDP/src/

export PYTHONPATH="$PYTHONPATH:/fs/cliplab/software/dumbo/python-2.7/lib/python2.7/site-packages"

export PYTHONPATH="$PYTHONPATH:/chomes/jbg/nltk_svn"
export PYTHONPATH="$PYTHONPATH:/fs/clip-software/protobuf-2.3.0b/python/lib/python2.6/site-packages"

export PYTHONPATH="$PYTHONPATH:$HOME/Workspace/topicmod/lib/python_lib"
export PYTHONPATH="$PYTHONPATH:/fs/cliplab/software/protobuf-2.3.0b/python/lib/python2.6/site-packages"
export PYTHONPATH="$PYTHONPATH:/fs/cliplab/software/PyStemmer/`arch`/lib/python2.5/site-packages/"
export PYTHONPATH="$PYTHONPATH:/fs/cliplab/software/pyeditdistance/`arch`/lib/python2.5/site-packages/"

CLIPCONTRIB="/fs/clip-software"
export NLTK_DATA=${CLIPCONTRIB}/nltk-2.0.1rc3-data
SCRATCH_DIRECTORY='/fs/clip-scratch/zhaike/PyHDP'
mkdir $SCRATCH_DIRECTORY

SET_PARAMETER

SnapshotInterval=100

mkdir $SCRATCH_DIRECTORY/output/$CorpusName

$PYTHON_COMMAND -O -um hdp.launch \
    --input_directory=$SCRATCH_DIRECTORY/input/$CorpusName \
    --output_directory=$SCRATCH_DIRECTORY/output/ \
    --training_iteration=$TrainingIteration \
    --snapshot_interval=$SnapshotInterval \
    --alpha_alpha=$AlphaAlpha \
    --alpha_gamma=$AlphaGamma \
    --alpha_eta=$AlphaEta \
    --split_merge_heuristics=$SMH \
    --split_proposal=$SP \
    --merge_proposal=$MP \
    > $SCRATCH_DIRECTORY/output/$CorpusName/T$TrainingIteration\-aa$AlphaAlpha\-ag$AlphaGamma-ae$AlphaEta$PostFix\-init30.out
#    > $SCRATCH_DIRECTORY/output/$CorpusName/T$TrainingIteration\-aa$AlphaAlpha\-ag$AlphaGamma\-smh$SMH\-sp$SP\-mp$MP\.out