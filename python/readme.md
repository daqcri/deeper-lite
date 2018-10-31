# DeepER-Lite Python

This folder contains a PyTorch implementation of DeepER-Lite and the LSH based blocking code. Please refer to the src folder for the source code. 
For convenience, we also have included 4 processed benchmark datasets Cora,  DBLP-ACM,  DBLP-Scholar and  Fodors-Zagat in the folder BenchmarkDatasets. 

Please contact Saravanan Thirumuruganathan or Mourad Ouzzani for any clarifications or assistance with code.

## DeepER-Lite for New Datasets

DeepER-Lite requires some preliminary configurations for setting up new datasets. 
For most of the benchmark datasets used in Entity Resolution, this has already been completed.
For a new dataset, you would have to do the following one-time setup:

1. Update the DATASET\_ROOT setting in configs.py to the folder in which the datasets are stored.
2. Add an entry in config.py about the new dataset. It already has entries for benchmark datasets.
3. Add a function for blocking in blocking\_utils.py . Functions already exist for known benchmark datasets.
4. Call the function save\_candset\_wrapper in blocking\_utils.py. It will create a processed version of the candidate set. For benchmark datasets, this is already done.
5. Call the function split\_dataset\_by\_ratio to generate training, validation and test files. For the benchmark datasets, it is already done. A sample invocation is given in process\_dataset\_sim.py 

## Running DeepER-Lite

1. In deeper\_lite\_sim.py, call train and test with appropriate parameters. Sample invocation for benchmark datasets is given already. This runs the model on a similarity vector of size 2\*m where m is the number of aligned attributes. We use cosine similarity for computing similarity between aligned attributes.
2. Alternatively, you can run deeper\_lite\_abs\_diff.py that runs a model that uses absolute vector difference. This will create a vector of size 300\*m where 300 is the size of embedding. 

## Blocking with LSH

1. Take a look at lsh\_blocking.py .
2. Note that in the paper, we used Magellan's blocking function to ensure fair comparison with it.  Hence, the lsh code is a bit disconnected from main deeper code. We tested the blocking of our methods independently. 
3. The code assumes that the word embeddings are available in a torch file. You can replace that function for other sources.

