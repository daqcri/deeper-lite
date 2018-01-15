'''
Created on Nov 19, 2017

@author: me
'''
import itertools
import argparse
import csv
import random
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--first_table')
parser.add_argument('--second_table')
parser.add_argument('--pred_file')
parser.add_argument('--num_pairs')
parser.add_argument('--num_parallels')


args = parser.parse_args()

first_file = open( args.first_table, encoding='utf-8',errors='ignore')
reader = csv.reader(first_file)
next(reader, None) 

first_ids = [line[0] for line in reader]
    
second_file = open( args.second_table, encoding='utf-8',errors='ignore')
reader = csv.reader(second_file)
next(reader, None) 

second_ids = [line[0] for line in reader]
    
num_pairs = int(args.num_pairs)	

counter = 0

pairs = list(itertools.product(first_ids, second_ids))

if num_pairs == -1:
	num_pairs = len(pairs)

random.shuffle(pairs)

pairs_len = num_pairs
num_parallels = int(args.num_parallels)
batch_size = math.floor(pairs_len/num_parallels)

start_indices = [0]
end_indices = [batch_size]



for i in range(num_parallels-1):
	start_indices.append(end_indices[len(end_indices)-1]+1)
	end_indices.append(min(start_indices[len(start_indices)-1] + batch_size,pairs_len-1))


for i in range(num_parallels):
	product_file = open(os.path.splitext(args.pred_file)[0] + "_" + str(i) + ".csv",'w')
	writer = csv.writer(product_file)
	for j in range(start_indices[i],end_indices[i]):
	    if counter < num_pairs:
	    	writer.writerow(pairs[j])
	    	counter = counter + 1
	product_file.close()
