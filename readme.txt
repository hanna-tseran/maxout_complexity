* All the requirements can be installed with pip using the command
	pip install 'library-name'

* Run one of the files with entry points with Python in the command line as
	python filename.py

* Two files with an entry point:
	1. regions_and_db_main.py demonstrates main routines at initialization
		- counts linear regions approximately and exactly
		- counts linear pieces in the decision boundary
		- computes theoretical formulas as a reference
		- plots linear regions in a 2D input space
		- plots decision boundary in a 2D input space
		* The output should be:
			Number of linear regions: 216
			Number of linear pieces in the decision boundary: 30
			Approximate number of linear regions: 199
			-----------------------------
			Upper bound on the expected number using full formula. Number of linear regions: 107528470. Number of linear pieces in the decision boundary: 2688211
			Asymptotic with K. Number of linear regions: 800. Number of linear pieces in the decision boundary: 40
			Asymptotic without K. Number of linear regions: 200. Number of linear pieces in the decision boundary: 20
			-----------------------------
			Plotted regions and the decision boundary. The results are in the "images" folder.
		* Two images that will be created are in the 'images' folder and have names 'initialization regions.png' and
			'initialization decision boundary.png'

	2. training_main.py trains the network and obtains the results of interest before and during training
		- trains a network on the MNIST dataset using one of the initializations
		- counts exactly linear regions, pieces in the decision boundary and plots the regions and the decision boundary
			in a slice determined by three data points before and during training.
		* The output should be:
			before training: 111 linear regions; 20 linear pieces in the decision boundary
			Epoch 1. Training loss: 0.8642288140142396. Accuracy: 0.8697
			Epoch 2. Training loss: 0.38970992890502343. Accuracy: 0.9079
			Epoch 3. Training loss: 0.3113546740970632. Accuracy: 0.9193
			Epoch 4. Training loss: 0.269193919800492. Accuracy: 0.926
			Epoch 5. Training loss: 0.24027930916563026. Accuracy: 0.9321
			after 5 epochs: 178 linear regions; 56 linear pieces in the decision boundary
			Epoch 6. Training loss: 0.2202531310842871. Accuracy: 0.9357
			Epoch 7. Training loss: 0.20586375998599188. Accuracy: 0.9397
			Epoch 8. Training loss: 0.19543696281466402. Accuracy: 0.9421
			Epoch 9. Training loss: 0.18715284791773062. Accuracy: 0.9414
			Epoch 10. Training loss: 0.17996366347458317. Accuracy: 0.9418
			after 10 epochs: 193 linear regions; 49 linear pieces in the decision boundary
		* Six images that will be created are in the 'images' folder and have names
			'before training regions.png', 'before training decision boundary.png',
			'after 5 epochs regions.png', 'after 5 epochs decision boundary.png',
			'after 10 epochs regions.png', 'after 10 epochs decision boundary.png'

* Edit parameters at the beginning of the 'main' function to get results for different parameters.

* The images will appear in the 'images' folder.

* Functions performing counting are in the Network class in network.py, and initializations are in init_method.py.

* This code submission contains only the main functions since the code for the experiments has cluster-specific routines
	and, moreover, cannot be run locally because of the resource requirements.