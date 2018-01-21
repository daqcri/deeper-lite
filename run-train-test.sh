#!/bin/bash
set -e

if [ $# -lt 8 ]; then
	echo "USAGE $0 <dataset-name> <first-table-name> <second-table-name> <num-table-fields> <do-sample> <cosine-sim-neg-threshold> <op-mode> <do-plot>"
	echo " "
	echo "Run this from DeepER-Lite root only"
	echo " "
	echo "e.g. bash run.sh Amazon-GoogleProducts Amazon GoogleProducts 4 0.2 train_test yes"
	echo " "
	exit 1
fi
DATASET="$1"
FIRST_TABLE=$2										#name only of second table (withoud extension)
SECOND_TABLE=$3										#name only of second table (without extension)
NUM_TABLE_FIELDS=$4									#how many columns/fields your tables have (excluding the id field)
SAMPLE=$5											#sample train/dev/test
COSINE_NEG_THRESHOLD=$6								#drop pairs with worse similarity than threshold
OP_MODE=$7                     						#[train_test | test]
PLOT=$8												#plot training and dev accuracies during training

DATA_DIR=$PWD/data
DATASET_DIR=$DATA_DIR/$DATASET

echo $DATA_SET_DIR
#these are defaults, you may leave them in peace
NEGATIVE_MAPPING_FILE=$DATASET_DIR/$DATASET'_negativeMapping.csv'
PERFECT_MAPPING_TRAIN_FILE=$DATASET_DIR/$DATASET'_perfectMappingTraining.csv'
PERFECT_MAPPING_TEST_FILE=$DATASET_DIR/$DATASET'_perfectMappingTesting.csv'
PERFECT_MAPPING_DEV_FILE=$DATASET_DIR/$DATASET'_perfectMappingDev.csv'
NEGATIVE_MAPPING_TRAIN_FILE=$DATASET_DIR/$DATASET'_negativetMappingTraining.csv'
NEGATIVE_MAPPING_TEST_FILE=$DATASET_DIR/$DATASET'_negativeMappingTesting.csv'
NEGATIVE_MAPPING_DEV_FILE=$DATASET_DIR/$DATASET'_negativeMappingDev.csv'
OUTOUT_DIR=$DATASET_DIR/output
mkdir -p $OUTOUT_DIR

#Sampling Params
SEED=31
RATIO_NEG_TO_POS_TYPE_1=1  						#e.g. 2 neg examples for each pos example where the table one record of a random pos is part of the tuple
RATIO_NEG_TO_POS_TYPE_2=1							#e.g. 2 neg examples for each pos example where the table two record of a random pos is part of the tuple
RATIO_NEG_TO_POS_TYPE_3=1							#e.g. 2 neg examples for each pos example where both records are not part of any pos example

PERCENT_NEG_TRAIN=0.75				#off all samples negative examples for training, what ratio do you want to keep (class balancing)

TRAIN_RATIO=0.25
DEV_RATIO=0.25
TEST_RATIO=0.5

#Training Params
THREADS=1 											#cpu threads
NOISE_DATA=no 										#nosie training data to test model robustness
NOISE_RATIO=0 										#how much noise [0 - 1]
RECOMPUTE_FEATURES=yes 								#after the first run features are saved, you are welcome to recompute them though
HIDDEN_X=2                                          #size of hidden Layer = HIDDEN_X x SIZE_OF_INPUT
LEARNING_RATE=1e-2
LEARNING_RATE_DROP_THRESHOLD=5e-3
LEARNING_RATE_DROP_CHECK_EPOCHSS=10
LEARNING_RATE_DROP_RATIO=0.1
BATCH_SIZE=10
EPOCHS=20


#SAMPLING
if [ "$SAMPLE" = "yes" ]
then
	echo 'Sampling ...'
   	mono Sampler.exe  		$DATASET_DIR/$DATASET"_perfectMapping.csv" \
						  	$NEGATIVE_MAPPING_FILE \
							$PERFECT_MAPPING_TRAIN_FILE \
							$PERFECT_MAPPING_TEST_FILE \
							$PERFECT_MAPPING_DEV_FILE \
							$NEGATIVE_MAPPING_TRAIN_FILE \
							$NEGATIVE_MAPPING_DEV_FILE \
							$NEGATIVE_MAPPING_TEST_FILE \
							"$DATASET_DIR/$FIRST_TABLE.csv" \
							"$DATASET_DIR/$SECOND_TABLE.csv" \
							$TRAIN_RATIO \
							$DEV_RATIO \
							$TEST_RATIO \
							$RATIO_NEG_TO_POS_TYPE_1 \
							$RATIO_NEG_TO_POS_TYPE_2 \
							$RATIO_NEG_TO_POS_TYPE_3 \
							1 \
							$SEED
	echo 'Done Sampling'
else
	echo 'No Sampling was done'
fi

#Training & Testing
if [ "$OP_MODE" = "train_test" ]
then
    echo "Training & Testing ..."
elif [ "$OP_MODE" = "test" ]
then
    echo "Testing ..."
fi

th DeepER-Lite.lua 	-positivePairsTrainingFile $PERFECT_MAPPING_TRAIN_FILE \
				  	-negativePairsTrainingFile $NEGATIVE_MAPPING_TRAIN_FILE \
				  	-positivePairsDevFile  $PERFECT_MAPPING_DEV_FILE \
				  	-negativePairsDevFile  $NEGATIVE_MAPPING_DEV_FILE \
				  	-positivePairsTestingFile  $PERFECT_MAPPING_TEST_FILE \
				  	-negativePairsTestingFile  $NEGATIVE_MAPPING_TEST_FILE \
				  	-positivePairsTrainingFileBin $PERFECT_MAPPING_TRAIN_FILE'.t7' \
				  	-negativePairsTrainingFileBin $NEGATIVE_MAPPING_TRAIN_FILE'.t7' \
				  	-positivePairsDevFileBin $PERFECT_MAPPING_DEV_FILE'.t7' \
				  	-negativePairsDevFileBin $NEGATIVE_MAPPING_DEV_FILE'.t7' \
				  	-positivePairsTestingFileBin $PERFECT_MAPPING_TEST_FILE'.t7' \
				  	-negativePairsTestingFileBin $NEGATIVE_MAPPING_TEST_FILE'.t7' \
				  	-testMapFileBin  $OUTOUT_DIR/testMap.t7 \
				  	-debug_file_path $OUTOUT_DIR/debug.txt \
				  	-test_predictions_file_path $OUTOUT_DIR/test_predictions.txt \
				  	-perf_file_path $OUTOUT_DIR/perf.txt \
				  	-threshold_file_path $OUTOUT_DIR/threshold.txt \
				  	-firstDataFile "$DATASET_DIR/$FIRST_TABLE.csv" \
				  	-secondDataFile "$DATASET_DIR/$SECOND_TABLE.csv" \
				  	-numTableFields $NUM_TABLE_FIELDS \
					-preTrainedModel glove.840B.300d \
					-embeddingSize 300 \
					-seed $SEED \
					-threads $THREADS \
					-model  mlp \
					-loss nll \
					-save $DATASET_DIR/results \
					-plot $PLOT \
					-optimization Adam \
					-hiddenX $HIDDEN_X \
					-learningRate $LEARNING_RATE \
					-learningRateDropThreshold $LEARNING_RATE_DROP_THRESHOLD \
					-learningRateDropCheckEpochs $LEARNING_RATE_DROP_CHECK_EPOCHSS \
					-learningRateDropRatio $LEARNING_RATE_DROP_RATIO \
					-batchSize $BATCH_SIZE \
					-weightDecay 1e-4 \
					-momentum 0.95 \
					-maxIter $EPOCHS \
					-type float \
					-minfreq 1 \
					-percent_neg_train $PERCENT_NEG_TRAIN \
					-simMeasure  cosineDiff \
					-computeFeatures $RECOMPUTE_FEATURES \
					-noiseFlipLabels  $NOISE_DATA \
					-noiseFlipLabelsRatio $NOISE_RATIO \
					-threshold $COSINE_NEG_THRESHOLD \
					-opMode $OP_MODE
