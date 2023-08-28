Spatio-Temporal Dynamic DeepHit Implementation with PyTorch
This repository provides an implementation of the Dynamic DeepHit model with PyTorch for survival analysis on PBC2 dataset.
It adds subject information with a GAT comparing the Dynamic DeepHit
It also adds temporal information relation before the RNN

Article:
Title: "Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks Based on Longitudinal Data"
Authors: Changhee Lee, Jinsung Yoon, Mihaela van der Schaar
Reference: C. Lee, J. Yoon, M. van der Schaar, "Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks Based on Longitudinal Data," IEEE Transactions on Biomedical Engineering (TBME). 2020
Paper: https://ieeexplore.ieee.org/document/8681104


Data Structure:
The data is expected to be in the format specific to the PBC2 mode. If a different data mode is used, ensure to modify the dataset import mechanism.
The key data structures include:
data: Matrix containing features.
time: Vector containing time to event or censoring.
label: Vector containing event indicators.

Model Configuration:
The model's hyperparameters and settings are stored in the new_parser dictionary and taken from Tensorflow implementation : https://github.com/chl8856/Dynamic-DeepHit


The main function is defined as training and evaluation:
-Training the Dynamic DeepHit model involves:
Initializing the model with the specified input dimensions and network settings.
Training the model on the training dataset.
Using a learning rate scheduler to adjust the learning rate based on loss.
Validating the model periodically to monitor performance and save the best model.
Applying early stopping if there's no improvement in the validation score after a specified number of epochs.

-Model evaluation includes:
Loading the best saved model.
Generating risk predictions for the test data at specified prediction and evaluation times.
Calculating performance metrics like the c-index and brier score for each combination of prediction and evaluation times.


How to Use:
Clone the repository.
Ensure all dependencies are installed with the requirement_cuda.txt file.
Modify the data importing mechanism if using a dataset other than PBC2.
Run the main function : the training function to train the model and the eval_model function to get the c-index and Brier score.
To see if the model is acceptable, the best one was saved on the name of copy_saved_model.pt and the performance can be shown with the Evaluation function.

Tips:
If another dataset is used, change the prediction and evaluation time to be in accordance with the dataset.
The best model is already saved, if it is changed a copy is in the copy_saved_model.pt file
