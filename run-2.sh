#!/bin/bash

set -e
# . /home/da/torch/install/bin/torch-activate
#Path Params: Edit these for your dataset please
#to train this model, you need 3 files, The first table, the second table and the perfect mapping
DATA_DIR=$PWD/data
if [ $# -lt 7 ]; then
	echo "USAGE $0 <Dataset> <first-table> <second-table> <num_table_fields> <prediction_file> <num_pairs> <num_parallels>"
	echo "Dataset should be relative to data directory"
	exit 1
fi
DATA_SET="$1"
FIRST_TABLE_FILE=$DATA_DIR/$DATA_SET/$2
SECOND_TABLE_FILE=$DATA_DIR/$DATA_SET/$3
NUM_TABLE_FIELDS=$4 			#how many columns/fields your tables have
PRED_FILE=$DATA_DIR/$DATA_SET/$5
NUM_PAIRS=$6
NUM_PARALLELS=$7
OUTOUT_DIR=$DATA_DIR/$DATA_SET/output
mkdir -p $OUTOUT_DIR




SEED=11
#Training Params
THREADS=1 					#cpu threads, DeepER-Lite is all done on cpus
RECOMPUTE_FEATURES=yes 		#after the first run features are saved, you are welcome to recompute them though

echo 'Generating Products ...'
python3 GenerateData.py --first_table $FIRST_TABLE_FILE \
						--second_table $SECOND_TABLE_FILE \
						--pred_file $PRED_FILE \
						--num_pairs $NUM_PAIRS \
						--num_parallels $NUM_PARALLELS
echo 'Done Generating Products ...'


echo "Prediction Mode ..."

START=0
END=$((NUM_PARALLELS - 1))

for (( c=$START; c<=$END; c++ ))
do
	echo ${PRED_FILE%.*}_$c.csv
	echo $OUTOUT_DIR/predictions_$c.txt
	th DeepER-Lite-2.lua -predPairsFile  ${PRED_FILE%.*}_$c.csv \
				  	 -predPairsFileBin ${PRED_FILE%.*}_$c.csv'.t7' \
				  	 -predMapFileBin  $OUTOUT_DIR/predMap.t7 \
				  	 -predictions_file_path $OUTOUT_DIR/predictions_$c.txt \
				  	 -firstDataFile $FIRST_TABLE_FILE \
				  	 -secondDataFile $SECOND_TABLE_FILE \
				  	 -numTableFields $NUM_TABLE_FIELDS \
					 -preTrainedModel glove.840B.300d \
					 -embeddingSize 300 \
					 -seed $SEED \
					 -threads $THREADS \
					 -save $DATA_DIR/$DATA_SET/results \
					 -type float \
					 -simMeasure  cosineDiff \
					 -threshold_file_path $OUTOUT_DIR/threshold.txt \
					 -computeFeatures $RECOMPUTE_FEATURES
done
exit
