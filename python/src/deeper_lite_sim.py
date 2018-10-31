#GiG.

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from sklearn.metrics import f1_score
import os
import random
import numpy as np
import pandas as pd

import process_dataset_sim as process_dataset

#DL Specific configs
BATCH_SIZE = 16
MAX_EPOCHS = 32
LEARNING_RATE = 0.001
BETAS = (0.9, 0.99)
EPSILON = 1e-9
RANDOM_STATE = 12345
HIDDEN_X = 2
MODEL_FILE_NAME = "best_validation_model_params.torch"

def get_deeper_lite_model_sim(num_attributes):

    #If there are K input attributes, Deeper Lite  has 2K features : 1 each for cosine distance and normed abs distance
    #Hidden_X is a multiplicative factor controlling the size of hidden layer.
    deeper_lite_model = nn.Sequential(
        nn.Linear(2 * num_attributes, HIDDEN_X * num_attributes),
        nn.ReLU(),
        nn.Linear(HIDDEN_X * num_attributes, HIDDEN_X * num_attributes),
        nn.ReLU(),
        nn.Linear(HIDDEN_X * num_attributes, HIDDEN_X * num_attributes),
        nn.ReLU(),
        nn.Linear(HIDDEN_X * num_attributes, 2),
    )

    return deeper_lite_model

#Assumes that the train and validation files are in the same folder as dataset_name
def train(dataset_name, train_file_name, validation_file_name, model_fn):
    train_features, train_labels = process_dataset.get_features_and_labels(dataset_name, train_file_name)
    validation_features, validation_labels = process_dataset.get_features_and_labels(dataset_name, validation_file_name)

    #Hack: Assumes that for deeper lite num_features = 2 * num_attributes
    num_attributes = train_features.shape[1] / 2
    model = model_fn(num_attributes)

    train_dataset = Data.TensorDataset(train_features, train_labels)
    #Allows us to read the dataset in batches
    training_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPSILON)
    criterion = nn.CrossEntropyLoss()


    #For reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    best_validation_f1_score = 0.0
    best_model_so_far = None
    model_file_name_path = os.path.join( process_dataset.get_folder_to_persist_model(dataset_name) , MODEL_FILE_NAME)

    for epoch in range(MAX_EPOCHS):
        for batch_idx, (train_features, train_labels) in enumerate(training_loader):
            optimizer.zero_grad()
            output = model(train_features)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()

        training_f1_score = compute_scores(output, train_labels)
        with torch.no_grad():
            validation_output = model(validation_features)
            validation_f1_score = compute_scores(validation_output, validation_labels)
            if validation_f1_score > best_validation_f1_score:
                best_model_so_far = model.state_dict()
                best_validation_f1_score = validation_f1_score

    torch.save(best_model_so_far, model_file_name_path)
    print "Curr Val F1, Best Val F1 ", validation_f1_score, best_validation_f1_score
    return best_model_so_far

def test(dataset_name, test_file_name, test_output_file_name, model_fn):
    test_features, test_labels = process_dataset.get_features_and_labels(dataset_name, test_file_name)
    #Hack: Assumes that for deeper lite num_features = 2 * num_attributes
    num_attributes = test_features.shape[1] / 2
    model = model_fn(num_attributes)

    folder_path = process_dataset.get_folder_to_persist_model(dataset_name)
    model_file_name_path = os.path.join( folder_path, MODEL_FILE_NAME)
    model.load_state_dict(torch.load(model_file_name_path))
    model.eval()

    predictions = model(test_features)
    #Uncomment the following lines to get the score
    #testing_f1_score = compute_scores(predictions, test_labels)
    #print "Testing F1 ", testing_f1_score

    prediction_as_numpy = torch.max(predictions, 1)[1].data.numpy()

    #Store output
    test_df = pd.read_csv( os.path.join(folder_path, test_file_name), encoding="utf-8" )
    test_df["gold"] = prediction_as_numpy
    test_df.to_csv(os.path.join(folder_path, test_output_file_name), encoding="utf8", index=False)

def compute_scores(predicted, actual):
    #Convert from cross entropy output to actual 0/1 predictions
    predicted = torch.max(predicted, 1)[1].data

    #Convert to numpy format
    predicted_numpy = predicted.numpy()
    actual_numpy = actual.numpy()

    #Print performance measures
    return f1_score(actual_numpy, predicted_numpy)

if __name__ == "__main__":
    train("Fodors_Zagat", "train.csv", "validation.csv", get_deeper_lite_model_sim)
    test("Fodors_Zagat", "test.csv", "test_predictions.csv", get_deeper_lite_model_sim)

    train("Cora", "train.csv", "validation.csv", get_deeper_lite_model_sim)
    test("Cora", "test.csv", "test_predictions.csv", get_deeper_lite_model_sim)

    train("DBLP_ACM", "train.csv", "validation.csv", get_deeper_lite_model_sim)
    test("DBLP_ACM", "test.csv", "test_predictions.csv", get_deeper_lite_model_sim)

    train("DBLP_Scholar", "train.csv", "validation.csv", get_deeper_lite_model_sim)
    test("DBLP_Scholar", "test.csv", "test_predictions.csv", get_deeper_lite_model_sim)
