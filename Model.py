import torch  # Importing the PyTorch library for deep learning
import torch.nn as nn # Import nn for layers
import torch.nn.init as init # Import the initialization module from PyTorch to set initial weights
_EPSILON = 1e-08 # Define a small constant to avoid division

from torch_geometric.nn import GATConv # Importing the PyTorch Geometric library for Graph

# Function to determine the length of a sequence (how many time steps are used in the sequence)
def get_seq_length(sequence):
    # Determine the absolute maximum values along the sequence and sign them (i.e., 1 for positive, 0 for zero, -1 for negative)
    used = torch.sign(torch.max(torch.abs(sequence), 2)[0])
    # Sum the used tensor to get the length for each sequence
    tmp_length = torch.sum(used, 1)
    # Convert the computed lengths to integers
    tmp_length = tmp_length.int()
    # Return the sequence lengths
    return tmp_length

# Fully connected network with activation function options
class FC_net(nn.Module):
    # Initialization function for the FC_net module
    def __init__(self, input_size, output_size,hidden_size, dropout,activ_out , nlayers=2, activation = 'ReLU'):
        super(FC_net, self).__init__()         # Initialize the superclass (nn.Module)
        # Activation function selection
        if activation == 'ReLU':
            self.activ = nn.ReLU()
        elif activation == 'Tanh':
            self.activ  = nn.Tanh()
        # Output activation function selection
        if activ_out=='ReLU':
            self.activ_out=nn.ReLU()
        else:
            self.activ_out=activ_out
        # Dropout rate assignment
        self.dropout = dropout
        # Number of layers assignment
        self.nlayers=nlayers
        # Define the first fully connected layer
        self.FC_1=nn.Linear(input_size, hidden_size)
        # Initialize the weights of the first fully connected layer using Xavier Uniform Initialization
        init.xavier_uniform_(self.FC_1.weight)
        # Define the second fully connected layer
        self.FC_2 = nn.Linear(hidden_size, output_size)
        # Initialize the weights of the second fully connected layer using Xavier Uniform Initialization
        init.xavier_uniform_(self.FC_2.weight)

    # Forward pass function for the FC_net module
    def forward(self,x):
        # Loop through the defined number of layers
        for i in range(self.nlayers):
            if i!=self.nlayers-1: # If not the last layer
                out0=self.FC_1(x) # Pass the input through the first FC layer
                out1=self.activ(out0) # Apply the activation function
                out3=nn.Dropout(p= self.dropout)(out1) # Apply dropout for regularization
            else:
                out = self.FC_2(out3)# Pass the processed tensor through the second FC layer
                if not self.activ_out == None: # If there's an output activation function defined, apply it
                    out = self.activ_out(out)
        return out

# Gated Recurrent Unit (GRU) with Dropout
class GRU_With_Dropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, bias=True):
        super(GRU_With_Dropout, self).__init__() # Initialize the superclass (nn.Module)
        self.dropout = dropout # Dropout rate assignment

        self.gru_cell_1 = nn.GRUCell(input_size, hidden_size, bias) # Define the first GRU cell
        init.xavier_uniform_(self.gru_cell_1.weight_hh) # Initialize the hidden weights of the first GRU cell using Xavier Uniform Initialization
        init.xavier_uniform_(self.gru_cell_1.weight_ih) # Initialize the input weights of the first GRU cell using Xavier Uniform Initialization
        # Clip the gradients of the parameters of gru_cell_1 to prevent exploding gradients, with a maximum norm of 1.
        torch.nn.utils.clip_grad_norm_(self.gru_cell_1.parameters(), max_norm=1, error_if_nonfinite=True)

        self.gru_cell_2 = nn.GRUCell(hidden_size, hidden_size, bias)# Define the second GRU cell
        init.xavier_uniform_(self.gru_cell_2.weight_hh) # Initialize the hidden weights of the second GRU cell using Xavier Uniform Initialization
        init.xavier_uniform_(self.gru_cell_2.weight_ih) # Initialize the input weights of the second GRU cell using Xavier Uniform Initialization
        # Clip the gradients of the parameters of gru_cell_2 to prevent exploding gradients, with a maximum norm of 1.
        torch.nn.utils.clip_grad_norm_(self.gru_cell_2.parameters(), max_norm=1, error_if_nonfinite=True)

        # Define a fully connected network to be used within this GRU network
        self.fc_net = FC_net(input_size=hidden_size*2+(int((input_size/3)*2)-2),output_size=1,hidden_size=hidden_size,dropout=dropout,activ_out=None,nlayers=2,activation='Tanh')

    # Forward pass function for the GRU_With_Dropout module
    def forward(self, inputs, all_last):
        # Initialize hidden states to 0 for the GRU cells
        gru1_hidden_state = torch.zeros((inputs.shape[0], 100)).to(inputs.device)
        gru2_hidden_state = torch.zeros((inputs.shape[0], 100)).to(inputs.device)
        # Lists to store hidden state and attention outputs from each timestep
        gru_outputs = []
        att_output = []
        # Loop through each timestep in the input sequence
        for t in range(inputs.size(1)):
            input = nn.Dropout(p=self.dropout)(inputs[:, t, :])  # Apply dropout to the current timestep input
            gru1_hidden_state_ = self.gru_cell_1(input, gru1_hidden_state) # Pass the processed input through the first GRU cell
            gru1_hidden_state = nn.Dropout(p=self.dropout)(gru1_hidden_state_)  # Apply dropout to the first GRU cell's output
            gru1_output = nn.Dropout(p=self.dropout)(gru1_hidden_state) # Store the output of the first GRU cell after applying dropout

            gru2_hidden_state_ = self.gru_cell_2(gru1_output, gru2_hidden_state) # Pass the output of the first GRU cell through the second GRU cell
            gru2_hidden_state = nn.Dropout(p=self.dropout)(gru2_hidden_state_)   # Apply dropout to the second GRU cell's output
            tmp_h = torch.cat([gru1_hidden_state, gru2_hidden_state], dim=1) # Concatenate the hidden states from both GRU cells
            e_ = self.fc_net(torch.cat([tmp_h,all_last],dim=1)) # Pass the concatenated tensor through the fully connected network for attention
            e=torch.exp(e_)# Apply the exponential function to the output of the FC network to create softmax
            # Append the concatenated hidden states and the attention outputs to their respective lists
            gru_outputs.append(tmp_h)
            att_output.append(e)
        # Convert lists of GRU outputs and attention outputs to tensors
        gru_output = torch.stack(gru_outputs, dim=1)
        att_output = torch.stack(att_output, dim=1).squeeze(2)
        # Return the GRU outputs and attention outputs
        return gru_output,att_output





# Define the main DynamicDeepHit model
class DynamicDeepHitTorch(nn.Module):
    # Initialization function for the DynamicDeepHit module
    def __init__(self, input_dim, output_dim,network_settings, risks,optimizer='Adam'):
        super(DynamicDeepHitTorch, self).__init__()
        # Store model input and output dimensions and hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.risks = risks
        # Extract hyperparameters from network settings
        self.h_dim1 = network_settings['h_dim_RNN']
        self.h_dim2 = network_settings['h_dim_FC']
        self.num_layers_RNN = network_settings['num_layers_RNN']
        self.num_layers_ATT = network_settings['num_layers_ATT']
        self.num_layers_CS = network_settings['num_layers_CS']
        self.RNN_type = network_settings['RNN_type']
        self.FC_active_fn = network_settings['FC_active_fn']
        self.RNN_active_fn = network_settings['RNN_active_fn']
        self.keep_prob =1- network_settings['keep_prob']
        self.num_Event = network_settings['num_Event']
        self.num_Category = network_settings['num_Category']

        # RNN layer with dropout
        self.rnn = GRU_With_Dropout(input_size=input_dim*3,
            hidden_size=self.h_dim1,dropout=self.keep_prob)

        # Longitudinal prediction layer
        self.longitudinal_prediction = nn.Linear(in_features=self.h_dim2*2,out_features=input_dim)
        init.xavier_uniform_(self.longitudinal_prediction.weight) # Initialize the weights of this layer

        # Cause-specific prediction layer
        self.combine_inputs = nn.Linear((input_dim - 1) + self.h_dim2 * 2, self.h_dim2)
        init.xavier_uniform_(self.combine_inputs.weight) # Initialize the weights of this layer

        # Initialize multiple cause-specific networks
        self.cause_specific = []
        for r in range(self.risks):
            self.cause_specific.append(
                FC_net(input_size=self.h_dim2, output_size=self.h_dim2, activ_out=self.FC_active_fn,
                       nlayers=self.num_layers_CS, hidden_size=self.h_dim2, dropout=self.keep_prob))
        self.cause_specific = nn.ModuleList(self.cause_specific)

        # Final linear layer
        self.out_linear = nn.Linear(self.h_dim2 * 2, self.output_dim * self.risks)
        init.xavier_uniform_(self.out_linear.weight)  # Initialize the weights of this layer
        # Probability computation using Softmax
        self.soft = nn.Softmax(dim=-1)  # On all observed output

    def forward(self, x, x_mi):  # x represent the data and x_mi the mask data
        # Determine the device to operate on (either CPU or GPU)
        if x.is_cuda:
            device = x.get_device()
        else:
            device = torch.device("cpu")

        # Compute sequence lengths for each patient to mask padding
        seq_length = get_seq_length(x).to(device)
        max_length = x.shape[1]
        feature_length = x.shape[2]
        # Create a tensor ranging from 0 to max_length
        tmp_range = torch.unsqueeze(torch.arange(0, max_length, 1), dim=0).to(device)
        # Create masks based on the sequence lengths
        rnn_mask1 = torch.le(tmp_range, torch.unsqueeze(seq_length - 1, dim=1))  # For loss function
        rnn_mask2 = torch.eq(tmp_range, torch.unsqueeze(seq_length - 1, dim=1))  # True for the last measurement

        # Compute history and the last measurement
        x_last = torch.sum(torch.unsqueeze(rnn_mask2, dim=2).repeat(1, 1, feature_length) * x,
                           dim=1)  # Sum over time, since all other time stamps are 0
        x_last = x_last[:, 1:]  # Remove delta of last measurement
        x_hist = x * (~torch.unsqueeze(rnn_mask2, dim=2).repeat(1, 1,
                                                                feature_length))  # Since all other time stamps are 0 and measurements are 0-padded
        x_hist = x_hist[:, :max_length - 1, :]  # Only take history up to the last time step

        x_mi_last = torch.sum(torch.unsqueeze(rnn_mask2, dim=2).repeat(1, 1, feature_length) * x_mi,
                              dim=1)  # Sum over time, since all other time stamps are 0
        x_mi_last = x_mi_last[:, 1:]  # Remove delta of last measurement
        x_mi_hist = x_mi * (~torch.unsqueeze(rnn_mask2, dim=2).repeat(1, 1,
                                                                      feature_length))  # Since all other time stamps are 0 and measurements are 0-padded
        x_mi_hist = x_mi_hist[:, :max_length - 1, :]  # Only take history up to the last time step

        all_last = torch.cat([x_last, x_mi_last], dim=1).to(device)
        all_hist = torch.cat([x_hist, x_mi_hist], dim=2).to(device)

        # Compute mask for attention of ej
        rnn_mask_att = torch.sum(x_hist, dim=2) != 0

        # Put tensor on the device
        x_last = x_last.to(device)
        # x_mi_last = x_mi_last.to(device)

        # Compute the graph attention mechanism
        # Initialize the adjacency matrix
        rows, cols = torch.meshgrid(torch.arange(x_hist.shape[0]), torch.arange(x_hist.shape[0]))
        edge_index = torch.stack((rows.flatten(), cols.flatten())).to(device)
        # Initialize the Gat layer
        gat_layer=GATConv(in_channels=x_hist.shape[2], out_channels=x_hist.shape[2]).to(device)

        # Compute the aggregation S
        S = gat_layer(x_hist.reshape(-1, x_hist.shape[2]), edge_index).reshape(x_hist.shape[0], x_hist.shape[1], -1)

        S=(S - S.min()) / (S.max() - S.min()) # BETTER THAN THE OTHER
        all_hist=torch.concat([all_hist,S],dim=2) # Concatenate history and neighborhood information

        # Compute hidden states and attention using the RNN
        hidden, att_weight = self.rnn(all_hist, all_last)

        # Compute the longitudinal prediction
        y_mean = self.longitudinal_prediction(hidden)
        y_std = torch.exp(self.longitudinal_prediction(hidden))
        epsilon = torch.randn(*(y_mean.shape[0], y_mean.shape[1], y_mean.shape[2])).to(device)
        y = y_mean + y_std * epsilon

        # Compute attention and normalize
        ej = att_weight.mul(rnn_mask_att)
        aj = torch.div(ej, torch.sum(ej) + _EPSILON)

        # Compute the context vector
        outputs = []
        attention = aj.unsqueeze(2).repeat(1, 1, hidden.size(2))
        c = torch.sum(attention.mul(hidden), dim=1)
        # Concatenate outputs from different risk networks
        SB_out__ = torch.cat([x_last, c], dim=1)
        SB_out_ = self.combine_inputs(SB_out__)
        SB_out = nn.Dropout(p=self.keep_prob)(SB_out_)
        for cs_nn in self.cause_specific:
            outputs.append(cs_nn(SB_out))

        # Soft max for probability distribution
        output0 = torch.cat(outputs, dim=1)
        output1 = nn.Dropout(p=self.keep_prob)(output0)
        output2 = self.out_linear(output1)
        output3 = self.soft(output2)
        # Rechange the size (batch,event,times)
        output = output3.view(-1, self.num_Event, self.num_Category)
        return y, output, rnn_mask1


