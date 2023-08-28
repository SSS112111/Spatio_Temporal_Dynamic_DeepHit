_EPSILON = 1e-08 # Define a small constant to avoid division


import numpy as np  # Importing the numpy library for array and matrix operations
import pandas as pd  # Importing the pandas library for data manipulation and analysis
import os  # Importing the os library for interacting with the operating system
import torch  # Importing the PyTorch library for deep learning


from sklearn.model_selection import train_test_split  # Importing function for splitting data
from tqdm import tqdm  # Importing tqdm for progress bars
import import_data as impt  # Import custom module for data importing
from Model import DynamicDeepHitTorch  # Importing the Dynamic Deep Hit model implemented in PyTorch
import time as TIME  # Importing the time module for timing purposes
import torch.optim as optim  # Importing the optim module for optimization algorithms
import losses  # Import custom loss functions module
from torch.utils.data import DataLoader  # DataLoader for batch processing
from utils_eval import c_index, brier_score, f_get_risk_predictions, save_logging  # Importing utility functions for evaluation
import wandb  # Importing Weights & Biases for experiment tracking
wandb.login  # Logging into Weights & Biases


# wandb.init(project="DDH_model",config={"epochs":300,"batch_size":32 })


# Define and initialize the Dynamic Deep Hit model settings
data_mode = 'PBC2'
seed = 1234

# Define the dataset parameters and import the dataset
'''
    num_Category            = max event/censoring time * 1.2
    num_Event               = number of events i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension including delta (1 + num_features)
    x_dim_cont              = dim of continuous features
    x_dim_bin               = dim of binary features
    mask1, mask2, mask3     = used for cause-specific network (FCNet structure)
'''

if data_mode == 'PBC2':
    (x_dim, x_dim_cont, x_dim_bin), (data, time, label), (mask1, mask2, mask3), (data_mi) = impt.import_dataset(
        norm_mode='standard')

    # This must be changed depending on the datasets, prediction/evaluation times of interest
    pred_time = [52, 3 * 52, 5 * 52]  # prediction time (in months)
    eval_time = [12, 36, 60, 120]  # months evaluation time (for C-index and Brier-Score)
else:
    print('ERROR:  DATA_MODE NOT FOUND !!!')

_, num_Event, num_Category = np.shape(mask1)  # dim of mask3: [subj, Num_Event, Num_Category]
max_length = np.shape(data)[1]

# Create directory for saving results
file_path = '{}'.format(data_mode)

if not os.path.exists(file_path):
    os.makedirs(file_path)

## Set Hyper-Parameters

new_parser = {'mb_size': 32,

              'iteration_burn_in': 3000,
              'iteration': 25000,

              'keep_prob': 0.6,
              'lr_train': 1e-2,

              'h_dim_RNN': 100,
              'h_dim_FC': 100,
              'num_layers_RNN': 2,
              'num_layers_ATT': 2,
              'num_layers_CS': 2,

              'RNN_type': 'GRU',  # {'LSTM', 'GRU'}

              'FC_active_fn':'ReLU',
              'RNN_active_fn': 'tanh',

              'reg_W': 1e-5,
              'reg_W_out': 0.,

              'alpha': 1.0,
              'beta': 0.1,
              'gamma': 1.0
              }

# Set the input dimensions
input_dims = {'x_dim': x_dim,
              'x_dim_cont': x_dim_cont,
              'x_dim_bin': x_dim_bin,
              'num_Event': num_Event,
              'num_Category': num_Category,
              'max_length': max_length}

# Set the Network hyperparameters
network_settings = {'h_dim_RNN': new_parser['h_dim_RNN'],
                    'h_dim_FC': new_parser['h_dim_FC'],
                    'num_layers_RNN': new_parser['num_layers_RNN'],
                    'num_layers_ATT': new_parser['num_layers_ATT'],
                    'num_layers_CS': new_parser['num_layers_CS'],
                    'RNN_type': new_parser['RNN_type'],
                    'FC_active_fn': new_parser['FC_active_fn'],
                    'RNN_active_fn': new_parser['RNN_active_fn'],
                    # 'initial_W'         : torch.nn.init.xavier_uniform_(),#tf.contrib.layers.xavier_initializer(),

                    'num_Event':num_Event,
                    'num_Category':num_Category,

                    'reg_W': new_parser['reg_W'],
                    'reg_W_out': new_parser['reg_W_out'],
                    'keep_prob': new_parser['keep_prob']
                    }
# More configurations for implementation
mb_size = new_parser['mb_size']
iteration = new_parser['iteration']
iteration_burn_in = new_parser['iteration_burn_in']

keep_prob = new_parser['keep_prob']
lr_train = new_parser['lr_train']

alpha = new_parser['alpha']
beta = new_parser['beta']
gamma = new_parser['gamma']

# Save the hyperparameter
log_name = file_path + '/hyperparameters_log.txt'
save_logging(new_parser, log_name)

# Split dataset into training, validation, and test sets : Test = 20% / Val = 0.2*0.8=16% / Train = 0.8*0.8=64%
(tr_data, te_data, tr_data_mi, te_data_mi, tr_time, te_time, tr_label, te_label,
 tr_mask1, te_mask1, tr_mask2, te_mask2, tr_mask3, te_mask3) = train_test_split(data, data_mi, time, label, mask1,
                                                                                mask2, mask3,
                                                                                test_size=0.2 ,random_state=seed)

(tr_data, va_data, tr_data_mi, va_data_mi, tr_time, va_time, tr_label, va_label,
 tr_mask1, va_mask1, tr_mask2, va_mask2, tr_mask3, va_mask3) = train_test_split(tr_data, tr_data_mi, tr_time, tr_label,
                                                                                tr_mask1, tr_mask2, tr_mask3,
                                                                                test_size=0.2 , random_state=seed)


# Detect if GPU is available and set device accordingly
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # availabe GPU
torch.cuda.set_device(device)  # set the device
torch.cuda.empty_cache()  # empty cash memory

# Create DataLoader for training data
train_data = impt.Dataset_all(tr_data, tr_data_mi, tr_time, tr_label,tr_mask1, tr_mask2, tr_mask3)
train_data_loader = DataLoader(train_data, batch_size=new_parser['mb_size'], shuffle=False)


# Training function
def train(train_data_loader):
    # Instantiate the model with given parameters and move it to the device
    model = DynamicDeepHitTorch(input_dim=x_dim, output_dim=num_Category, network_settings=network_settings,
                                risks=num_Event).to(device)
    # Set the model to training mode
    model.train()
    # Define number of training epochs
    nb_epoch = 1500  #new_parser['iteration']
    # Define the optimizer (Adam) with learning rate, weight decay settings
    optimizer = optim.Adam(model.parameters(), lr=new_parser['lr_train'],weight_decay=new_parser['reg_W'])
    # Initialize minimal validation c-index score to monitor progress
    min_valid = 0.5
    # Counter for early stopping
    early_stopping = 0
    # Learning rate scheduler to reduce learning rate on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5,
                                                     patience=10,
                                                     verbose=True, min_lr=1e-5)
    # Temporary validation score holder
    tmp_valid = 0
    # Loop over epochs with a progress bar
    with tqdm(range(nb_epoch)) as tdx:
        for epoch in tdx:
            # Initialize epoch loss
            epoch_loss = 0
            # Track start time of epoch for performance monitoring
            start = TIME.time()
            # Loop over batches in the training dataset
            for iter, (data_) in enumerate(train_data_loader):
                # Zero out gradients
                optimizer.zero_grad()

                # Get the data
                data = data_[0].to(device)
                data_mi = data_[1].to(device)
                time = data_[2].to(device)
                event = data_[3].to(device)
                mask1 = data_[4].to(device)
                mask2 = data_[5].to(device)
                mask3 = data_[6].to(device)

                # Forward pass: get predictions from model
                longitudinal_prediction, outcomes,rnn_mask1 = model.forward(data, data_mi)

                # Compute the loss
                loss = losses.total_loss(longitudinal_prediction, outcomes, data,data_mi, time, event,mask1, mask2, mask3,rnn_mask1,
                                         new_parser['alpha'], new_parser['beta'], new_parser['gamma'])
                # Perform backpropagation
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1,norm_type=2)
                # Update model weights
                optimizer.step()
                # Adjust learning rate based on loss
                scheduler.step(loss)
                # Accumulate epoch loss
                epoch_loss += loss.detach().item()


            # Calculate average epoch loss
            epoch_loss /= (iter + 1)
            # Display epoch details in progress bar

            tdx.set_postfix(time=TIME.time() - start,  epoch_loss=epoch_loss,lr=optimizer.param_groups[0]['lr'],tmp_valid=tmp_valid)

            # wandb.log({'loss': epoch_loss}) #Display loss on wandb
            # Validate model every 10 epochs
            if (epoch) % 10 == 0:
                # Get risk predictions for validation data
                risk_all = f_get_risk_predictions(model, va_data, va_data_mi, pred_time, eval_time, device)
                # Loop over prediction times
                for p, p_time in enumerate(pred_time):
                    pred_horizon = int(p_time)
                    val_result1 = np.zeros([num_Event, len(eval_time)])
                    # Loop over evaluation times
                    for t, t_time in enumerate(eval_time):
                        eval_horizon = int(t_time) + pred_horizon
                        for k in range(num_Event):
                            # Calculate c-index for the current prediction and evaluation times
                            val_result1[k, t] = c_index(risk_all[k][:, p, t], va_time,
                                                        (va_label[:, 0] == k + 1).astype(int),
                                                        eval_horizon)  # -1 for no event (not comparable)
                    # Aggregate results across prediction times
                    if p == 0:
                        val_final1 = val_result1
                    else:
                        val_final1 = np.append(val_final1, val_result1, axis=0)
                # Get average c-index
                tmp_valid = np.mean(val_final1)
                # Save model if current validation score is the best so far
                if tmp_valid > min_valid:
                    min_valid = tmp_valid
                    torch.save(model, 'saved_model.pt')
                    print('model saved', 'c_index = ' + str(min_valid))
                    early_stopping=0
                else:
                    early_stopping +=1
                # If no improvement after 10 epochs, stop training
                if early_stopping==10:
                    break

# Define the evaluation function
def eval_model():
    # Load the best model
    model=torch.load('saved_model.pt')
    # Set the model to evaluation mode
    model.eval()
    # Get risk predictions for test data
    risk_all = f_get_risk_predictions(model, te_data, te_data_mi, pred_time, eval_time,device)
    # Initialize result holders
    for p, p_time in enumerate(pred_time):
        pred_horizon = int(p_time)
        result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])
        # Loop over evaluation times
        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time) + pred_horizon
            for k in range(num_Event):
                # Calculate c-index and brier score
                result1[k, t] = c_index(risk_all[k][:, p, t], te_time, (te_label[:, 0] == k + 1).astype(int),
                                        eval_horizon)  # -1 for no event (not comparable)
                result2[k, t] = brier_score(risk_all[k][:, p, t], te_time, (te_label[:, 0] == k + 1).astype(int),
                                            eval_horizon)  # -1 for no event (not comparable)
        # Aggregate results across prediction times
        if p == 0:
            final1, final2 = result1, result2
        else:
            final1, final2 = np.append(final1, result1, axis=0), np.append(final2, result2, axis=0)
    # Prepare headers for result tables
    row_header = []
    for p_time in pred_time:
        for t in range(num_Event):
            row_header.append('pred_time {}: event_{}'.format(p_time, t + 1))

    col_header = []
    for t_time in eval_time:
        col_header.append('eval_time {}'.format(t_time))

    # Convert results to pandas dataframes
    df1 = pd.DataFrame(final1, index=row_header, columns=col_header)# c-index result
    df2 = pd.DataFrame(final2, index=row_header, columns=col_header)# brier-score result

    # Display results
    print('========================================================')
    print('--------------------------------------------------------')
    print('- C-INDEX: ')
    print(df1)
    print('--------------------------------------------------------')
    print('- BRIER-SCORE: ')
    print(df2)
    print('========================================================')
    print("Mean C index for event 1 = " , final1[::2,:].mean())
    print("Mean C index for event 2 = ", final1[3::2, :].mean())



train(train_data_loader) # Train the model
eval_model() # Eval the model
