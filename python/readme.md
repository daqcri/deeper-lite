One time setup for new datasets:
	- Update the DATASET_ROOT setting in configs.py
	- Add an entry in config.py about the new dataset. It already has some entries for benchmark datasets
	- Add a function for blocking in blocking_utils.py . Functions already exist for known benchmark datasets.
	- Call the function save_candset_wrapper in blocking_utils. It will create a processed version of the candidate set. For benchmark datasets, this is already done.
	- Call the function split_dataset_by_ratio to generate training, validation and test files. For the benchmark datasets, it is already done. A sample invocation is given in process_dataset_sim.py 

Running DeepER:
	- In deeper_lite_sim.py, call train and test with appropriate parameters. Sample invocation for benchmark datasets is given already. This runs the model on a similarity vector of size 2*m where m is the number of aligned attributes. We use cosine similarity for computing similarity between aligned attributes.
	- Alternatively, you can run deeper_lite_abs_diff.py that runs a model that uses absolute vector difference. This will create a vector of size 300*m where 300 is the size of embedding. 

LSH:
	- Take a look at lsh_blocking.py .
	- Note that in the paper, we used Magellan's blocking function to ensure fair comparison with it.  Hence, the lsh code is a bit disconnected from main deeper code. We tested the blocking of our methods independently. 
	- The code assumes that the word embeddings are available in a torch file. You can replace that function for other sources.

