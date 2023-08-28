import torch  # Importing the PyTorch library for deep learning
_EPSILON = 1e-08 # Define a small constant to avoid division

# Function to compute the loss based on Log Likelihood
def loss_Log_Likelihood(outcomes,mask1,mask2,event):
    I_1 = torch.sign(event) # Compute indicator for events (1 if event, 0 otherwise)
    # Compute the denominator for the log likelihood, clamping values for numerical stability
    denom_ = 1 - torch.sum(torch.sum(mask1 * outcomes, dim=2), dim=1, keepdim=True)
    denom = torch.clamp(denom_, _EPSILON, 1. - _EPSILON)

    # Compute two terms of the loss, conditioned on the event indicator
    tmp1_ = torch.sum(torch.sum(mask2 * outcomes, dim=2), dim=1, keepdim=True)
    tmp1 = I_1 * torch.log(tmp1_ / denom)

    tmp2_ = torch.sum(torch.sum(mask2 * outcomes, dim=2), dim=1, keepdim=True)
    tmp2 = (1. - I_1) * torch.log(tmp2_ / denom)

    # Combine and average the two terms to compute the final loss
    loss=-torch.mean(tmp1 + tmp2)
    return loss

# Function to compute the ranking loss
def ranking_loss(outcomes,time, event,mask3):
    sigma1 = torch.tensor(0.1, dtype=torch.float32)
    eta = []
    # Compute ranking loss for each distinct event
    for e in range(int(max(event))):
        one_vector = torch.ones_like(time, dtype=torch.float32)
        I_2 = torch.eq(event, e + 1).float()  # Indicator for specific event
        I_2 = torch.diag(I_2.squeeze())
        tmp_e = outcomes[:, e, :].view(-1,mask3.shape[1])  # Get event specific joint prob.

        # Compute risk scores
        R = torch.matmul(tmp_e, mask3.t())  # no need to divide by each individual dominator
        # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

        diag_R =  R.diag().view(-1, 1)
        R = torch.matmul(one_vector, diag_R.t()) - R  # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = R.t()  # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        # Create mask for time comparison
        T = torch.nn.functional.relu(
            torch.sign(torch.matmul(one_vector, time.t()) - torch.matmul(time, one_vector.t())))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
        T = torch.matmul(I_2, T)  # Only remains T_{ij}=1 when event occured for subject i
        tmp_eta = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)
        eta.append(tmp_eta)

    # Combine the computed loss terms for all events
    eta = torch.stack(eta, dim=1)  # stack referenced on subjects
    eta = torch.mean(eta.view(-1, int(max(event))), dim=1, keepdim=True)

    loss=torch.sum(eta)

    return loss  # sum over num_Events

# Function to compute the longitudinal loss
def longitudinal_loss(y, x,x_mi,rnn_mask1):
    # Obtain data starting from time t=2 onward
    tmp_x = x[:, 1:, :]  # (t=2 ~ M)
    tmp_mi = x_mi[:, 1:, :]  # (t=2 ~ M)
    max_length=x.shape[1]

    # Mask for history up to last measurement
    tmp_mask1 = rnn_mask1.unsqueeze(2).repeat(1, 1,x.shape[2])  # for history (1...J-1)
    tmp_mask1 = tmp_mask1[:, :(max_length - 1), :]
    # Compute the squared difference between predicted and actual values, weighted by the mask and missing indicator
    zeta = torch.mean(torch.sum(tmp_mask1 * (1. - tmp_mi) * torch.pow(y - tmp_x, 2),
                                dim=1))  # loss calculated for selected features.

    return zeta

# Function to compute the total loss combining the above three losses
def total_loss(longitudinal_prediction, outcomes, data,data_mi, time, event,mask1, mask2, mask3,rnn_mask1, alpha, beta, gamma):

    L1=loss_Log_Likelihood(outcomes,mask1,mask2,event)
    L2=ranking_loss(outcomes,time, event,mask3)
    L3 = longitudinal_loss(longitudinal_prediction, data,data_mi,rnn_mask1)

    # Compute the total loss as a weighted sum of the three losses
    loss=alpha * L1 + beta * L2 + gamma * L3

    return loss