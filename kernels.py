import numpy as np

def linear_kernel(X_train, X_valid, scale=True, mode="train"):
    """
    Computes the linear kernel for X_train (n_train x n_train) as well as the kernel for X_valid (n_valid x n_train) on the chosen RHKS.
    "train" mode outputs the training and validation kernels
    "test" mode only outputs the training kernel
    """
    
    if scale:
        X_tr = (X_train-X_train.mean(axis=0)) / X_train.std(axis=0)
        X_va = (X_valid-X_train.mean(axis=0)) / X_train.std(axis=0)
        
        K_va = X_va @ X_tr.T
        
        if mode == "test":
            return(K_va)
        
        K_tr = X_tr @ X_tr.T
        
    else:
        K_va = X_valid @ X_train.T
        
        if mode == "test":
            return(K_va)
        
        K_tr = X_train @ X_train.T
        
    return(K_tr, K_va)

def gaussian_kernel(X_train, X_valid, sigma=None, scale=True, scale_sigma=True, mode="train"):
    
    n, p = X_train.shape
    
    if scale:
        X_tr = (X_train-X_train.mean(axis=0)) / X_train.std(axis=0)
        X_va = (X_valid-X_train.mean(axis=0)) / X_train.std(axis=0)
    if scale_sigma:
        sigma = p
        
    K_va = np.linalg.norm(X_va[:, None, ...] - X_tr[None, ...], axis=-1)**2
    K_va = np.exp((-K_va)/(sigma))
    
    if mode=="test":
        return(K_va)
    
    else:
        
        K_tr = ((X_tr[:, :, None] - X_tr[:, :, None].T) ** 2).sum(1)
        K_tr = np.exp((-K_tr)/(sigma))
        return(K_tr, K_va)
    
def polynomial_kernel(X_train, X_valid, d=3, c=1, scale=True, mode="train"):
    
    # k(x,y) = (<x,y> + c)**d
    if scale:
        
        X_tr = (X_train-X_train.mean(axis=0)) / X_train.std(axis=0)
        X_va = (X_valid-X_train.mean(axis=0)) / X_train.std(axis=0)
        
        K_va = X_va @ X_tr.T + c
        K_va = K_va**d
        
        if mode=="test":
            return(K_va)
        
        K_tr = X_tr @ X_tr.T + c
        K_tr = K_tr**d
    
        
    else:
        K_va = X_valid @ X_train.T + c
        K_va = K_va**d
        
        if mode=="test":
            return(K_va)
        
        K_tr = X_train @ X_train.T + c
        K_tr = K_tr**d
        
    return(K_tr, K_va)
