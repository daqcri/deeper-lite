'''
Created on Nov 19, 2017

@author: me
'''
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--preds_dir_path')
parser.add_argument('--all_preds_file_path')
parser.add_argument('--perfect_file_path')
parser.add_argument('--is_paranthesized')

args = parser.parse_args()

pred_file_counter = 0
for file in os.listdir(args.preds_dir_path):
	if file.endswith(".csv"):
		pred_lines = open(os.path.join(args.preds_dir_path, file),'r').read().split('\n')
		if args.is_paranthesized == 'no':
			pred_lines = [line.replace('"', '') for line in pred_lines]
		if (pred_file_counter == 0):
			pred_ids = set(pred_lines)
		else:
			pred_ids = pred_ids.union(set(pred_lines))
		pred_file_counter = pred_file_counter + 1



with open(args.all_preds_file_path, 'w') as all_preds_file:
	for pred_id in pred_ids:
	  all_preds_file.write("%s\n" % pred_id)

perfect_lines = open(args.perfect_file_path,'r').read().split('\n')
# perfect_lines = [line.replace('"', '') for line in perfect_lines]
perfect_ids = set(perfect_lines)

true_positives = pred_ids.intersection(perfect_ids)
false_positives = pred_ids.difference(perfect_ids)
false_negatives = perfect_ids.difference(pred_ids)

print('true positives:    ' + str(len(true_positives)))
print('false positives:   ' + str(len(false_positives)))
print('false negatives:   ' + str(len(false_negatives)))
