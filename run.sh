#!/bin/bash
set -e
# . /root/torch/install/bin/torch-activate
#Path Params: Edit these for your dataset please
#to train this model, you need 3 files, The first table, the second table and the perfect mapping
DATA_DIR=$PWD/data
if [ $# -lt 6 ]; then
	echo "USAGE $0 <Dataset> <first-table> <second-table> <perfect-mappings-file> <num_table_fields> <cosine_neg_threshold>"
	echo "Dataset should be relative to data directory"
	exit 1
fi
DATA_SET="$1"
FIRST_TABLE_FILE=$DATA_DIR/$DATA_SET/$2
SECOND_TABLE_FILE=$DATA_DIR/$DATA_SET/$3
PERFECT_MAPPING_FILE=$DATA_DIR/$DATA_SET/$4
NUM_TABLE_FIELDS=$5			#how many columns/fields your tables have
COSINE_NEG_THRESHOLD=$6
#these are defaults, you may leave them in peace
NEGATIVE_MAPPING_FILE=$DATA_DIR/$DATA_SET/$DATA_SET'_negativeMapping.csv'
PERFECT_MAPPING_TRAIN_FILE=$DATA_DIR/$DATA_SET/$DATA_SET'_perfectMappingTraining.csv'
PERFECT_MAPPING_TEST_FILE=$DATA_DIR/$DATA_SET/$DATA_SET'_perfectMappingTesting.csv'
PERFECT_MAPPING_DEV_FILE=$DATA_DIR/$DATA_SET/$DATA_SET'_perfectMappingDev.csv'
NEGATIVE_MAPPING_TRAIN_FILE=$DATA_DIR/$DATA_SET/$DATA_SET'_negativetMappingTraining.csv'
NEGATIVE_MAPPING_TEST_FILE=$DATA_DIR/$DATA_SET/$DATA_SET'_negativeMappingTesting.csv'
NEGATIVE_MAPPING_DEV_FILE=$DATA_DIR/$DATA_SET/$DATA_SET'_negativeMappingDev.csv'
OUTOUT_DIR=$DATA_DIR/$DATA_SET/output
mkdir -p $OUTOUT_DIR

#Sampling Params
SEED=11
SAMPLE=yes
NEGATIVE_RATIO_TYPE_1=10 	#where first part of sampled negative pair is part of a perfect pair
NEGATIVE_RATIO_TYPE_2=10	#where second part of sampled negative pair is part of a perfect pair
NEGATIVE_RATIO_TYPE_3=10	#where neither part of sampled negative pair is part of a perfect pair
TRAIN_RATIO=0.1
DEV_RATIO=0.1
TEST_RATIO=0.8
NEGATIVE_TRAIN_KEEP_RATIO=1 #do you really need all those negatives in training

#Training Params
THREADS=1 					#cpu threads, DeepER-Lite is all done on cpus
NOISE_DATA=no 				#nosie training data to test model robustness
NOISE_RATIO=0 				#how much noise [0 - 1]
OP_MODE=train_test 			#[train_test | test]
RECOMPUTE_FEATURES=yes 		#after the first run features are saved, you are welcome to recompute them though
PLOT=no					#plot training and dev accuracies during training

#SAMPLING
if [ "$SAMPLE" = "yes" ]
then
	echo 'Sampling ...'
   mono Sampler.exe  	$PERFECT_MAPPING_FILE \
						$NEGATIVE_MAPPING_FILE \
						$PERFECT_MAPPING_TRAIN_FILE \
						$PERFECT_MAPPING_TEST_FILE \
						$PERFECT_MAPPING_DEV_FILE \
						$NEGATIVE_MAPPING_TRAIN_FILE \
						$NEGATIVE_MAPPING_DEV_FILE \
						$NEGATIVE_MAPPING_TEST_FILE \
						$FIRST_TABLE_FILE \
						$SECOND_TABLE_FILE \
						$TRAIN_RATIO \
						$DEV_RATIO \
						$TEST_RATIO \
						$NEGATIVE_RATIO_TYPE_1 \
						$NEGATIVE_RATIO_TYPE_2 \
						$NEGATIVE_RATIO_TYPE_3 \
						$NEGATIVE_TRAIN_KEEP_RATIO \
						$SEED
	echo 'Done Sampling'
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
				  	-firstDataFile $FIRST_TABLE_FILE \
				  	-secondDataFile $SECOND_TABLE_FILE \
				  	-numTableFields $NUM_TABLE_FIELDS \
					-preTrainedModel glove.840B.300d \
					-embeddingSize 300 \
					-seed $SEED \
					-threads $THREADS \
					-model  mlp \
					-loss nll \
					-save $DATA_DIR/$DATA_SET/results \
					-plot $PLOT \
					-optimization Adam \
					-hiddenX 2 \
					-learningRate 1e-2 \
					-learningRateDropThreshold 5e-3 \
					-learningRateDropCheckEpochs 50 \
					-learningRateDropRatio 1e-1 \
					-batchSize 1 \
					-weightDecay 1e-4 \
					-momentum 0.95 \
					-maxIter 30 \
					-type float \
					-minfreq 1 \
					-percent_train 1 \
					-simMeasure  cosineDiff \
					-computeFeatures $RECOMPUTE_FEATURES \
					-noiseFlipLabels  $NOISE_DATA \
					-noiseFlipLabelsRatio $NOISE_RATIO \
					-threshold $COSINE_NEG_THRESHOLD \
					-opMode $OP_MODE
