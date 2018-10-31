#GiG.

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from sklearn.metrics import f1_score
import os
import random
import numpy as np

import process_dataset_abs_diff
import configs

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#DL Specific configs
BATCH_SIZE = 16
MAX_EPOCHS = 16
LEARNING_RATE = 0.001
BETAS = (0.9, 0.99)
EPSILON = 1e-9
RANDOM_STATE = 12345
MODEL_FILE_NAME = "best_validation_model_params_abs_diff.torch"

#Changed from 2 to 4
HIDDEN_X = 4

def get_deeper_lite_model(num_attributes):

    #If there are K input attributes, Deeper Lite  has 2K features : 1 each for cosine distance and normed abs distance
    #Hidden_X is a multiplicative factor controlling the size of hidden layer.
    deeper_lite_model = nn.Sequential(
        nn.Linear(configs.DR_DIMENSION * num_attributes, HIDDEN_X * num_attributes),
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
    train_features, train_labels = process_dataset_abs_diff.get_features_and_labels(dataset_name, train_file_name)
    validation_features, validation_labels = process_dataset_abs_diff.get_features_and_labels(dataset_name, validation_file_name)

    #Hack: Assumes that for deeper lite num_features = 2 * num_attributes
    num_attributes = train_features.shape[1] / configs.DR_DIMENSION
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
    model_file_name_path = os.path.join( process_dataset_abs_diff.get_folder_to_persist_model(dataset_name) , MODEL_FILE_NAME)

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

def test(dataset_name, test_file_name, model_fn):
    test_features, test_labels = process_dataset_abs_diff.get_features_and_labels(dataset_name, test_file_name)
    #Hack: Assumes that for deeper lite num_features = 2 * num_attributes
    num_attributes = test_features.shape[1] / 300
    model = model_fn(num_attributes)

    model_file_name_path = os.path.join( process_dataset_abs_diff.get_folder_to_persist_model(dataset_name) , MODEL_FILE_NAME)
    model.load_state_dict(torch.load(model_file_name_path))
    model.eval()

    output = model(test_features)
    testing_f1_score = compute_scores(output, test_labels)
    print "Testing F1 ", testing_f1_score

def train_sklearn_classifiers(dataset_name, train_file_name, validation_file_name):
    train_features, train_labels = process_dataset_abs_diff.get_features_and_labels(dataset_name, train_file_name)
    validation_features, validation_labels = process_dataset_abs_diff.get_features_and_labels(dataset_name, validation_file_name)

    models = []
    models.append(LogisticRegression(random_state=RANDOM_STATE))
    models.append(RandomForestClassifier(random_state=RANDOM_STATE))
    models.append(LinearSVC(random_state=RANDOM_STATE))

    for model in models:
        model.fit(train_features, train_labels)

    return models

def test_sklearn_classifiers(dataset_name, test_file_name, models):
    test_features, test_labels = process_dataset_abs_diff.get_features_and_labels(dataset_name, test_file_name)
    for model in models:
        predicted = model.predict(test_features)
        print model.__class__, f1_score(predicted, test_labels)


def compute_scores(predicted, actual):
    #Convert from cross entropy output to actual 0/1 predictions
    predicted = torch.max(predicted, 1)[1].data

    #Convert to numpy format
    predicted_numpy = predicted.numpy()
    actual_numpy = actual.numpy()

    #Print performance measures
    return f1_score(actual_numpy, predicted_numpy)

if __name__ == "__main__":
    #train("Fodors_Zagat", "train_30_20_50.csv", "validation_30_20_50.csv")
    #test("Fodors_Zagat", "test_30_20_50.csv")

    #train("Cora", "train_30_20_50.csv", "validation_30_20_50.csv")
    #test("Cora", "test_30_20_50.csv")

    #train("Amazon_GoogleProducts", "train_30_20_50.csv", "validation_30_20_50.csv")
    #test("Amazon_GoogleProducts", "test_30_20_50.csv")

    import configs
    for dataset_name in configs.er_dataset_details:
    #for dataset_name in ["Fodors_Zagat", "Cora", "Amazon_GoogleProducts"]:
        print dataset_name
        #train(dataset_name, "train_30_20_50.csv", "validation_30_20_50.csv", get_deeper_lite_model)
        models = train_sklearn_classifiers(dataset_name, "train_30_20_50.csv", "validation_30_20_50.csv")
        #test(dataset_name, "test_30_20_50.csv", get_deeper_lite_model)
        test_sklearn_classifiers(dataset_name, "test_30_20_50.csv", models)

