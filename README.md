# On the Expected Complexity of Maxout Networks

This repository is the official implementation of [On the Expected Complexity of Maxout Networks](anonymous). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running code

This repository contains implementation of the key routines used in the paper experiments.
Functions performing counting are in the Network class in network.py, and initializations are in init_method.py.
The images will appear in the 'images' folder.
We provide the scripts allowing to get results at initialization and during training.

### At initialization

To run:
```running at initialization
python regions_and_db_main.py
```

Demonstrates the main routines at initialization
 - counts linear regions approximately and exactly
 - counts exactly linear pieces in the decision boundary
 - computes theoretical formulas as a reference
 - plots linear regions in a 2D input space
 - plots decision boundary in a 2D input space
 
 The output should be:
 ```initialization output
Number of linear regions: 216
Number of linear pieces in the decision boundary: 30
Approximate number of linear regions: 199

Upper bound on the expected number using full formula. Number of linear regions: 107528470. Number of linear pieces in the decision boundary: 2688211
Asymptotic with K. Number of linear regions: 800. Number of linear pieces in the decision boundary: 40
Asymptotic without K. Number of linear regions: 200. Number of linear pieces in the decision boundary: 20

Plotted regions and the decision boundary. The results are in the "images" folder.
```
Two images that will be created are in the 'images' folder and have names 'initialization regions.png' and 'initialization decision boundary.png'

  
### During training

```running training
python training_main.py
```

Trains the network and obtains the results of interest before and during training
 - trains a network on the MNIST dataset using one of the initializations
 - counts exactly linear regions, pieces in the decision boundary and plots the regions and the decision boundary in a slice determined by three data points before and during training.

The output should be:
```training output
before training: 111 linear regions; 20 linear pieces in the decision boundary
Epoch 1. Training loss: 0.864. Accuracy: 0.87
Epoch 2. Training loss: 0.39. Accuracy: 0.908
Epoch 3. Training loss: 0.311. Accuracy: 0.919
Epoch 4. Training loss: 0.269. Accuracy: 0.926
Epoch 5. Training loss: 0.24. Accuracy: 0.932
after 5 epochs: 178 linear regions; 56 linear pieces in the decision boundary
Epoch 6. Training loss: 0.22. Accuracy: 0.936
Epoch 7. Training loss: 0.206. Accuracy: 0.94
Epoch 8. Training loss: 0.195. Accuracy: 0.942
Epoch 9. Training loss: 0.187. Accuracy: 0.941
Epoch 10. Training loss: 0.18. Accuracy: 0.942
after 10 epochs: 193 linear regions; 49 linear pieces in the decision boundary
```

Six images that will be created are in the 'images' folder and have names 'before training regions.png', 'before training decision boundary.png', 'after 5 epochs regions.png', 'after 5 epochs decision boundary.png', 'after 10 epochs regions.png', 'after 10 epochs decision boundary.png'

## Running with custom parameters
Edit parameters at the beginning of the 'main' function of a corresponding script to get results for different parameters.
