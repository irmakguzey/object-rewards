import numpy as np
import ot
import torch


def optimal_transport_plan(
    X,
    Y,
    cost_matrix,
    method="sinkhorn_gpu",
    niter=500,
    epsilon=0.01,
    exponential_weight_init=False,
    as_torch=True,
):
    if exponential_weight_init:
        a = 2 / 3  # We will be implmenting
        r = 1 / 3  # We are approximating these initial values - by looking at the plots
        N_x = X.shape[0]
        N_y = Y.shape[0]
        X_pot = [a * (r ** (N_x - n)) for n in range(N_x)]
        Y_pot = [a * (r ** (N_y - n)) for n in range(N_y)]
    else:
        X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
        Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])

    if as_torch:
        c_m = cost_matrix.data.detach().cpu().numpy()
    else:
        c_m = np.asarray(cost_matrix.data)
    transport_plan = ot.sinkhorn(X_pot, Y_pot, c_m, epsilon, numItermax=niter)
    if as_torch:
        transport_plan = torch.from_numpy(transport_plan).to(X.device)
        transport_plan.requires_grad = False
    return transport_plan


def cosine_distance(x, y, as_torch=True):
    if as_torch:
        C = torch.mm(x, y.T)
        x_norm = torch.norm(x, p=2, dim=1)
        y_norm = torch.norm(y, p=2, dim=1)
        x_n = x_norm.unsqueeze(1)
        y_n = y_norm.unsqueeze(1)
        norms = torch.mm(x_n, y_n.T)
    else:
        # Compute the dot product matrix
        C = np.dot(x, y.T)

        # Compute the norms of x and y
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)

        # Compute the norms matrix
        norms = np.dot(x_norm, y_norm.T)

    C = 1 - C / (norms + 1.0e-12)
    return C


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, 2))
    return c
