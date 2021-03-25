import numpy as np

def to_binary(y):
    """
    takes an array with values in [-1,1] and turns into 0 all the values that are below 0
    """
    y_ = np.zeros(y.shape[0])
    y_[y>0] = 1
    return(y_)

def error(y_true, y_pred):
    y_pred_ = to_binary(y_pred)
    return(1-np.mean(y_pred_ == y_true))

def no_weighted_mv(K_train, K_val, alphas, y_train, y_val, K_test):
    
    """
    INPUTS
    K_train is a list of training kernels
    K_val is a list of the respective validation kernels
    K_test is a list of the respective testing kernels
    alphas is a list containing the alpha vector found for each training kernel
    
    
    OUTPUTS
    Predictions for K_test
    """
    
    prob = 1/len(K_train)
    y_tr_pred = np.zeros(y_train.shape[0])
    y_val_pred = np.zeros(y_val.shape[0])
    y_te_pred = np.zeros(1000)
    
    for i in range(len(K_train)):
        
        y_tr_i= K_train[i] @ alphas[i]
        y_val_i= K_val[i] @ alphas[i]
        
        #err = error(y_train, y_tr_i)
        #if err == 0:
        #    err = 10
        #else:
        #    err = gamma * np.log((1-err)/err)
        
        #errors += [err]
        
        y_tr_pred += prob * y_tr_i
        y_val_pred += prob * y_val_i
        y_te_pred += prob * (K_test[i] @ alphas[i])
    
    #print("Assigned Weights : ", errors)
    print(f"Training score : {1 - error(y_train, y_tr_pred)}")
    print(f"Validation score : {1 - error(y_val, y_val_pred)}")
    
    return(y_te_pred)

def weighted_mv(K_train, K_val, alphas, y_train, y_val, K_test, gamma= 1/2 ):
    
    
    """
    INPUTS
    K_train is a list of training kernels
    K_val is a list of the respective validation kernels
    K_test is a list of the respective testing kernels
    alphas is a list containing the alpha vector found for each training kernel
    
    
    OUTPUTS
    Predictions for K_test
    """
    errors = []
    y_tr_pred = np.zeros(y_train.shape[0])
    y_val_pred = np.zeros(y_val.shape[0])
    y_te_pred = np.zeros(1000)
    
    for i in range(len(K_train)):
        
        y_tr_i= K_train[i] @ alphas[i]
        y_val_i= K_val[i] @ alphas[i]
        
        err = error(y_train, y_tr_i)
        if err == 0:
            err = 10
        else:
            err = gamma * np.log((1-err)/err)
        
        errors += [err]
        
        y_tr_pred += err * y_tr_i
        y_val_pred += err * y_val_i
        y_te_pred += err * (K_test[i] @ alphas[i])
    
    print("Assigned Weights : ", errors)
    print(f"Training score : {1 - error(y_train, y_tr_pred)}")
    print(f"Validation score : {1 - error(y_val, y_val_pred)}")
    return(y_te_pred)