#!/bin/bash

set -e
if [ $# -lt 8 ]; then
	echo "USAGE $0 <dataset-name> <first-table-name> <second-table-name> <num-table-fields> <pairs-file> <num-pairs> <num-pairs-files> <do-generate-pairs>"
	echo " "
	echo "Run this from DeepER-Lite root only"
	echo " "
	echo "e.g. bash run-pred.sh fodors-zagats fodors zagats 6 pairs_to_predict -1 1 yes"
	echo " "
	exit 1
fi

DATASET="$1"
FIRST_TABLE=$2										#name only of second table (withoud extension)
SECOND_TABLE=$3										#name only of second table (without extension)
NUM_TABLE_FIELDS=$4									#how many columns/fields your tables have (excluding the id field)
PRED_FILE=$5							
NUM_PAIRS=$6
NUM_FILES=$7
GENERATE_PAIRS=$8
DATA_DIR=$PWD/data
DATASET_DIR=$DATA_DIR/$DATASET
OUTOUT_DIR=$DATASET_DIR/output
mkdir -p $OUTOUT_DIR
PRED_FILE_PATH="$DATASET_DIR/$PRED_FILE.csv"
FIRST_TABLE_PATH="$DATASET_DIR/$FIRST_TABLE.csv"
SECOND_TABLE_PATH="$DATASET_DIR/$SECOND_TABLE.csv"
SEED=31
EMPTY_COSINE_PENALTY=0

#Training Params
THREADS=$NUM_FILES 									#cpu threads
RECOMPUTE_FEATURES=yes 								#after the first run features are saved, you are welcome to recompute them though


if [ "$GENERATE_PAIRS" = "yes" ]
then
	echo 'Generating Pairs ...'
	python3 GenerateData.py --first_table $FIRST_TABLE_PATH \
							--second_table $SECOND_TABLE_PATH \
							--pred_file $PRED_FILE_PATH \
							--num_pairs $NUM_PAIRS \
							--num_parallels $NUM_FILES
	echo 'Done Generating Pairs ...'
else
	echo "No Pairs Were Generated!"
fi

echo "Predicting Matches ..."

START=0
END=$((NUM_FILES - 1))

for (( c=$START; c<=$END; c++ ))
do
	echo ${PRED_FILE%.*}_$c.csv
	echo $OUTOUT_DIR/predictions_$c.txt
	th DeepER-Lite-Pred.lua -predPairsFile  ${PRED_FILE_PATH%.*}_$c.csv \
				  	 -predPairsFileBin ${PRED_FILE_PATH%.*}_$c.csv'.t7' \
				  	 -predMapFileBin  $OUTOUT_DIR/predMap.t7 \
				  	 -predictions_file_path $OUTOUT_DIR/predictions_$c.csv \
				  	 -firstDataFile $FIRST_TABLE_PATH \
				  	 -secondDataFile $SECOND_TABLE_PATH \
				  	 -numTableFields $NUM_TABLE_FIELDS \
					 -preTrainedModel glove.840B.300d \
					 -embeddingSize 300 \
					 -seed $SEED \
					 -threads $THREADS \
					 -save $DATASET_DIR/results \
					 -type float \
					 -simMeasure  cosineDiff \
					 -threshold_file_path $OUTOUT_DIR/threshold.txt \
					 -empty_cosine_penalty $EMPTY_COSINE_PENALTY \
					 -computeFeatures $RECOMPUTE_FEATURES
done
