#GiG.

#This file mostly utility functions to process dataset files to the expected data format

import configs
import blocking_utils

import csv
import os
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
import fastText
from scipy.spatial.distance import cosine
import torch

#This function splits an input file from a given path into three : train, validation and test
# then output files train.csv, validation.csv, test.csv in the folder folder_path
# conservatively, the split is done in a stratified manner by manually splitting data into duplicates and non duplicates
# this is relevant when we test very small number of training data

def split_dataset_by_ratio(folder_path, candset_ids_file_name, split_ratio=[0.3, 0.2, 0.5], label_field_name= 'gold', random_state=12345):
    df = pd.read_csv(os.path.join(folder_path, candset_ids_file_name), encoding="utf-8")
    duplicates_df = df[df[label_field_name] == 1]
    non_duplicates_df = df[df[label_field_name] == 0]

    train_duplicates, validation_duplicates, test_duplicates = local_train_validate_test_split(duplicates_df, split_ratio, random_state)
    train_non_duplicates, validation_non_duplicates, test_non_duplicates = local_train_validate_test_split(non_duplicates_df, split_ratio, random_state)

    #The last sample is to shuffle the data so that duplicates and non_duplicates mix
    train_df = pd.concat([train_duplicates, train_non_duplicates]).sample(frac=1)
    validation_df = pd.concat([validation_duplicates, validation_non_duplicates]).sample(frac=1)
    test_df = pd.concat([test_duplicates, test_non_duplicates]).sample(frac=1)

    #verify_split(df, train_df, validation_df, test_df, split_ratio, label_field_name)
    for partition_df, file_name in [(train_df, "train.csv"), (validation_df, "validation.csv"), (test_df, "test.csv")]:
        partition_df.to_csv(os.path.join(folder_path, file_name), encoding="utf-8", index=False)


def local_train_validate_test_split(df, split_ratio=[0.3, 0.2, 0.5], random_state=12345):
    np.random.seed(random_state)
    random_shuffle = np.random.permutation(df.index)
    num_tuples = len(df)

    train_end = int(num_tuples * split_ratio[0])
    validation_end = train_end + int(num_tuples * split_ratio[1])

    train_df = df.ix[random_shuffle[:train_end]]
    validation_df = df.ix[random_shuffle[train_end:validation_end]]
    test_df = df.ix[random_shuffle[validation_end:]]

    return train_df, validation_df, test_df

#Trivial manual validation to check correctness of the split
def verify_split(df, train_df, validation_df, test_df, split_ratio, label_field_name):
    num_duplicates_df = len(df[df[label_field_name] == 1])
    num_non_duplicates_df = len(df[df[label_field_name] == 0])

    for index, partition_df in enumerate([train_df, validation_df, test_df]):
        num_partition_duplicates = len(partition_df[partition_df[label_field_name] == 1])
        num_partition_non_duplicates = len(partition_df[partition_df[label_field_name] == 0])
        expected_duplicates, expected_non_duplicates = int(num_duplicates_df * split_ratio[index]), int(num_non_duplicates_df * split_ratio[index])
        actual_duplicates, actual_non_duplicates = num_partition_duplicates, num_partition_non_duplicates
        if actual_duplicates != expected_duplicates or actual_non_duplicates != expected_non_duplicates:
            print "Mismatch :", expected_duplicates, actual_duplicates, expected_non_duplicates, actual_non_duplicates

#Given a set of (ltable_id, rtable_id, gold) triples,
# construct the distributional similarity vector and store it
#First is the folder, second is the train|validation|test file, and the last two are the csv files of the left and right datasets
#Output file name is obtained from input_file_name and is put in the same folder as folder_path
# This is not very efficient - partially to avoid storing large intermediate matrices
def dataset_to_matrix(folder_path, input_file_name, ltable_file_name, rtable_file_name, output_file_name):
    ltable_df = pd.read_csv( os.path.join(folder_path, ltable_file_name), encoding="utf-8")
    rtable_df = pd.read_csv( os.path.join(folder_path, rtable_file_name), encoding="utf-8")

    candset_with_ids_df = pd.read_csv( os.path.join(folder_path, input_file_name), encoding="utf-8" )

    #Find common attributes of ltable and rtable
    common_attributes = ltable_df.columns.intersection(rtable_df.columns)
    #Only keep the common attributes
    ltable_df = ltable_df[common_attributes]
    rtable_df = rtable_df[common_attributes]

    #Assumption id column is titled as "id"
    #Remove this from nltk processing
    common_attributes = list(common_attributes.drop("id"))
    ltable_df = ltable_df.fillna(" ")
    rtable_df = rtable_df.fillna(" ")

    #This is a list of punctuations to remove with empty string
    replacement_list = get_replacement_list()

    for attribute in common_attributes:
        ltable_df[attribute] = ltable_df[attribute].replace(regex=replacement_list, value= " ")
        rtable_df[attribute] = rtable_df[attribute].replace(regex=replacement_list, value= " ")

    #Assumes key column name is id. Creates a dictionary where key is the id and the value is row index
    id_to_idx_dict_ltable = dict(zip(ltable_df.id, ltable_df.index))
    id_to_idx_dict_rtable = dict(zip(rtable_df.id, rtable_df.index))

    #First m attributes are for cosine diff and next m for norm of abs diff
    #The last one is for gold label
    num_attributes = len(common_attributes)
    dist_repr_similarity_matrix = np.zeros( (len(candset_with_ids_df), configs.DR_DIMENSION * num_attributes + 1), dtype=np.float32)

    #The following is an inefficient way to construct distributional similarity vectors
    # but avoids the need to create huge distributional representations for input vectors
    fasttext_model = load_fasttext_model()
    for row_index, row in candset_with_ids_df.iterrows():
        ltable_id, rtable_id, gold = row

        #Get the index of the current (ltable_id, rtable_id, gold) being processed
        ltable_index = id_to_idx_dict_ltable[ row['ltable_id'] ]
        rtable_index = id_to_idx_dict_rtable[ row['rtable_id'] ]

        #Get the corresponding rows
        ltable_row = ltable_df.iloc[ltable_index]
        rtable_row = rtable_df.iloc[rtable_index]

        for col_index, attribute in enumerate(common_attributes):
            abs_diff = compute_distance_abs_diff(fasttext_model, ltable_row[attribute], rtable_row[attribute])
            start_pos = col_index * configs.DR_DIMENSION
            end_pos = (col_index+1) * configs.DR_DIMENSION
            dist_repr_similarity_matrix[row_index][start_pos:end_pos] = abs_diff
            dist_repr_similarity_matrix[row_index][-1] = gold

    np.save(os.path.join(folder_path, output_file_name), dist_repr_similarity_matrix)

#This function takes two strings, converts to utf-8, computes their cosine and absolute error distance
def compute_distance(fasttext_model, ltable_str, rtable_str):
    if isinstance(ltable_str, basestring) == False:
        ltable_str = unicode(ltable_str)
    if isinstance(rtable_str, basestring) == False:
        rtable_str = unicode(rtable_str)

    lcol_dr = fasttext_model.get_sentence_vector(ltable_str)
    rcol_dr = fasttext_model.get_sentence_vector(rtable_str)

    #See the fillna command before for handling nulls
    if ltable_str == rtable_str and ltable_str == " ":
        #If both empty return lowest distance
        return 0.0, 0.0
    if ltable_str == " " and rtable_str != " ":
        return 1.0, 1.0
    if ltable_str != " " and rtable_str == " ":
        return 1.0, 1.0


    cosine_dist = cosine(lcol_dr, rcol_dr)
    normed_abs_dist = np.linalg.norm(np.abs(lcol_dr - rcol_dr))
    return cosine_dist, normed_abs_dist

#This function takes two strings, converts to utf-8, computes their cosine and absolute error distance
def compute_distance_abs_diff(fasttext_model, ltable_str, rtable_str):
    if isinstance(ltable_str, basestring) == False:
        ltable_str = unicode(ltable_str)
    if isinstance(rtable_str, basestring) == False:
        rtable_str = unicode(rtable_str)

    lcol_dr = fasttext_model.get_sentence_vector(ltable_str)
    rcol_dr = fasttext_model.get_sentence_vector(rtable_str)

    return np.abs(lcol_dr - rcol_dr)

#This is a helper function to create distributional similarity matrix for all datasets
# and calls the dist similarity computation for each of train, validation and test files
def compute_dist_similarity_matrix_wrapper():
    all_datasets = configs.er_dataset_details.keys()
    for dataset_name in all_datasets:

        dataset = configs.er_dataset_details[dataset_name]
        folder = dataset["dataset_folder_path"]
        input_file = "candset_ids_only.csv"
        ltable_file_name = dataset["ltable_file_name"]
        rtable_file_name = dataset["rtable_file_name"]

        #dataset_to_matrix("/Users/neo/Desktop/QCRI/DataCleaning/datasets/BenchmarkDatasets/Fodors_Zagat/", "candset_ids_only.csv", "fodors.csv", "zagats.csv")
        #dataset_to_matrix(folder, "candset_ids_only.csv", ltable_file, rtable_file)

        print "Processing ", dataset_name, " train.csv"
        dataset_to_matrix(folder, "train.csv", ltable_file_name, rtable_file_name)
        print "Processing ", dataset_name, " validation.csv"
        dataset_to_matrix(folder, "validation.csv", ltable_file_name, rtable_file_name)
        print "Processing ", dataset_name, " test.csv"
        dataset_to_matrix(folder, "test.csv", ltable_file_name, rtable_file_name)

def convert_csv_to_features(dataset_name, input_file_name):
    dataset = configs.er_dataset_details[dataset_name]
    folder_path = dataset["dataset_folder_path"]
    ltable_file_name = dataset["ltable_file_name"]
    rtable_file_name = dataset["rtable_file_name"]

    feature_file_name = input_file_name.replace(".csv", "_abs_diff.npy")

    #Check if the npy file already exists
    file_path = os.path.join(folder_path, feature_file_name)
    if os.path.exists(file_path):
        print "File {} already exists. Reusing it.".format(feature_file_name)
    else:
        print "File {} does not exist. Creating and persisting it.".format(feature_file_name)
        dataset_to_matrix(folder_path, input_file_name, ltable_file_name, rtable_file_name, feature_file_name)
    return np.load(file_path)

def get_features_and_labels(dataset_name, file_name):
    matrix = convert_csv_to_features(dataset_name, file_name)
    features, labels = matrix[:, :-1], matrix[:, -1]

    #Convert to torch format from numpy format
    features, labels = torch.from_numpy(features), torch.from_numpy(labels).type(torch.LongTensor)
    return features, labels


#Function to efficiently remove common punctuation and stopwords
def get_replacement_list():
    replacement_list = [r'\.', r"\'", r'\"', r'\(', '\)', r'\,', r'\&', r'\\', r'\/']
    #for stopword in stopwords.words('english'):
    #    replacement_list.append(r'\b%s\b' % stopword)
    return replacement_list

def load_fasttext_model():
    return fastText.load_model(configs.FASTTEXT_MODEL_PATH)

def get_folder_to_persist_model(dataset_name):
    dataset = configs.er_dataset_details[dataset_name]
    folder = dataset["dataset_folder_path"]
    return folder

if __name__ == "__main__":
    compute_dist_similarity_matrix_wrapper()
