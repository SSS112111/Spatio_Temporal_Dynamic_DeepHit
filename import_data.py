import pandas as pd # for data manipulation and analysis
import numpy as np  # for numerical operations
from torch.utils.data import Dataset# for creating custom datasets for PyTorch
import torch # Importing the PyTorch library for deep learning


# Function to normalize input data X based on a given normalization mode
def f_get_Normalization(X, norm_mode):
    # Retrieve the shape of the matrix (num_Patient rows and num_Feature columns)
    num_Patient, num_Feature = np.shape(X)
    # Standardize the data if the mode is 'standard'
    if norm_mode == 'standard':  # zero mean unit variance
        for j in range(num_Feature):
            if np.nanstd(X[:, j]) != 0:
                X[:, j] = (X[:, j] - np.nanmean(X[:, j])) / np.nanstd(X[:, j])
            else:
                X[:, j] = (X[:, j] - np.nanmean(X[:, j]))
    # Min-Max normalization if the mode is 'normal'
    elif norm_mode == 'normal':  # min-max normalization
        for j in range(num_Feature):
            X[:, j] = (X[:, j] - np.nanmin(X[:, j])) / (np.nanmax(X[:, j]) - np.nanmin(X[:, j]))
    else:
        print("INPUT MODE ERROR!")  # Print error message for unsupported normalization modes

    return X

# Function to generate the mask used in the loss calculation (for conditional probability)
def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask3 is required to get the contional probability (to calculate the denominator part)
        mask3 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category])  # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0] + 1)] = 1  # last measurement time

    return mask

# Function to generate the mask used in the loss calculation (log-likelihood loss)
def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category])  # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i, 0] != 0:  # not censored
            mask[i, int(label[i, 0] - 1), int(time[i, 0])] = 1
        else:  # label[i,2]==0: censored
            mask[i, :, int(time[i, 0] + 1):] = 1  # fill 1 until from the censoring time (to get 1 - \sum F)
    return mask

# Function to generate the mask used in the loss calculation (ranking loss)

def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category])  # for the first loss function
    if np.shape(meas_time):  # lonogitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0])  # last measurement time
            t2 = int(time[i, 0])  # censoring/event time
            mask[i, (t1 + 1):(t2 + 1)] = 1  # this excludes the last measurement time and includes the event time
    else:  # single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0])  # censoring/event time
            mask[i, :(t + 1)] = 1  # this excludes the last measurement time and includes the event time
    return mask


# Function to construct the dataset from a dataframe and a feature list
def f_construct_dataset(df, feat_list):
    '''
        id   : patient indicator
        tte  : time-to-event or time-to-censoring
            - must be synchronized based on the reference time
        times: time at which observations are measured
            - must be synchronized based on the reference time (i.e., times start from 0)
        label: event/censoring information
            - 0: censoring
            - 1: event type 1
            - 2: event type 2
            ...
    '''

    grouped = df.groupby(['id']) # Number of subject
    id_list = pd.unique(df['id']) # Number of measurement per subject
    max_meas =  np.max(grouped.size()) # Max measurement for each subject
    data = np.zeros([len(id_list), max_meas, len(feat_list) + 1]) # Get the data
    pat_info = np.zeros([len(id_list), 5]) # Get the label

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        pat_info[i, 4] = tmp.shape[0]  # number of measurement
        pat_info[i, 3] = np.max(tmp['times'])  # last measurement time
        pat_info[i, 2] = tmp['label'][0]  # cause
        pat_info[i, 1] = tmp['tte'][0]  # time_to_event
        pat_info[i, 0] = tmp['id'][0]
        data[i, :int(pat_info[i, 4]), 1:] = tmp[feat_list]
        data[i, :int(pat_info[i, 4] - 1), 0] = np.diff(tmp['times']) # add the time into the features

    return pat_info, data

# Imports and preprocesses the dataset.
def import_dataset(norm_mode='standard'):
    df_ = pd.read_csv('./data/pbc2_cleaned.csv') # Read the CSV file into a dataframe

    # Define lists for binary and continuous features
    bin_list = ['drug', 'sex', 'ascites', 'hepatomegaly', 'spiders']
    cont_list = ['age', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin',
                 'histologic']
    feat_list = cont_list + bin_list # Combination
    # Filter the dataframe with specific columns
    df_ = df_[['id', 'tte', 'times', 'label'] + feat_list]
    df_org_ = df_.copy(deep=True) # Create a copy of the original dataframe

    # Normalize continuous features
    df_[cont_list] = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    # Construct dataset for the normalized and original data
    pat_info, data = f_construct_dataset(df_, feat_list)
    _, data_org = f_construct_dataset(df_org_, feat_list)

    data_mi = np.zeros(np.shape(data)) # Initialize a zero matrix with shape of data for missing information
    # Indicate missing values
    data_mi[np.isnan(data)] = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)] = 0

    # Define dimensions for the data
    x_dim = np.shape(data)[2]  #  x_dim_cont + x_dim_bin + 1 (including delta)
    x_dim_cont = len(cont_list)
    x_dim_bin = len(bin_list)

    # Extract age at the last measurement, label, and event time information from pat_info
    last_meas = pat_info[:, [3]]  # pat_info[:, 3] contains age at the last measurement
    label = pat_info[:, [2]]  # two competing risks
    time = pat_info[:, [1]]  # age when event occurred

    # Determine the number of maximum time prediction
    num_Category = int(np.max(pat_info[:, 1]) * 1.2)  # 744*1.2=892  or specifically define larger than the max tte
    # Determine number of unique events
    num_Event = len(np.unique(label)) - 1

    # Convert multi-risk data to single risk if only one event type is present
    if num_Event == 1:
        label[np.where(label != 0)] = 1  # make single risk

    # Get masks for the data
    mask1 = f_get_fc_mask1(last_meas, num_Event, num_Category)
    mask2 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3 = f_get_fc_mask3(time, -1, num_Category)

    # Define data dimensions and data sets
    DIM = (x_dim, x_dim_cont, x_dim_bin)
    DATA = (data, time, label)
    MASK = (mask1, mask2, mask3)

    return DIM, DATA, MASK, data_mi


class Dataset_all(Dataset):
    def __init__(self, d1, d2, d3,d4,d5,d6,d7):
        self.d1 = torch.tensor(d1,dtype=torch.float32) # Data
        self.d2 = torch.tensor(d2,dtype=torch.float32) # Missing data
        self.d3 = torch.tensor(d3,dtype=torch.float32) # Time-to-event
        self.d4 = torch.tensor(d4, dtype=torch.float32)# Labeled event
        self.d5 = torch.tensor(d5, dtype=torch.float32)# Mask1
        self.d6 = torch.tensor(d6, dtype=torch.float32)# Mask2
        self.d7 = torch.tensor(d7, dtype=torch.float32)# Mask3

    def __len__(self):
        return (self.d1.shape[0]) # Get bumber of subject for batch

    # Get the item data needed
    def __getitem__(self, item):
        return self.d1[item, :, :],self.d2[item, :, :] ,self.d3[item, :],\
            self.d4[item, :],self.d5[item, :, :] ,self.d6[item, :, :] , self.d7[item, :]
