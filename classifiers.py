import numpy as np
from cvxopt import matrix, solvers


def accuracy(y_true,y_pred, mode='SVM'):
    n = y_true.shape[0]
    if mode == 'SVM':
        predictions = np.ones(n)
        predictions[y_pred < 0] = 0
    else:
        predictions = np.zeros(n)
        predictions[y_pred >= 0.5] = 1
    
    return np.sum(y_true == predictions) / n


def MSE(K, y, lambd, alpha):
    """
    Computes the penalized Mean Squared Error
    K : (n_samples * n)
    y : (n_samples)
    alpha : n
    lambda >0
    """
    n = y.shape[0]
    data_term = (np.linalg.norm(np.dot(K, alpha.reshape(-1,1)) - y)**2)/n
    reg_term = alpha @ K @ alpha
    return(data_term + lambd * reg_term)

def KRR(K, y, Kval, yval, lambd):
    """
    takes the kernel matrix as an input and computes the MSE and the predictions for each value in lambd (list)
    """
    assert K.shape[0] == y.shape[0]
    assert len(lambd) > 0
    n = K.shape[0]
    
    loss = []
    acc = []
    
    loss_val = []
    acc_val = []
    alphas = []
    
    for l in lambd:
        
        assert l >= 0
        # find the parameter alpha
        alpha = np.linalg.solve((K + l*n*np.eye(n)), y)
        # predict
        
        loss_lambda = MSE(K, y, l, alpha)
        acc_lambda = accuracy(y,K@alpha, mode="KRR")
        
        loss_lambdaval = MSE(Kval, yval, l, alpha)
        acc_lambdaval = accuracy(yval,Kval@alpha, mode="KRR")

        print(f"***********lambda = {l}***********")
        print(f"Training: loss = {loss_lambda:.4f}, accuracy = {acc_lambda:.6f}")
        print(f"Validation: loss = {loss_lambdaval:.4f}, accuracy = {acc_lambdaval:.6f}")
        
        loss += [loss_lambda]
        acc += [acc_lambda]
        
        loss_val += [loss_lambdaval]
        acc_val += [acc_lambdaval]
        
        
        alphas +=[alpha]
        
    return(alphas, loss, acc, loss_val, acc_val)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_loss(y_true, y_pred):
    n = y_true.shape[0]
    log_term = np.log(sigmoid(y_true*y_pred))
    return(-np.sum(log_term)/n)
    
def KLR(K, y, Kval, yval, lambd, maxIter = 100, tresh = 1e-8):
    
    # initialize the values
    assert K.shape[0] == y.shape[0]
    n = K.shape[0]
    
    y_ = np.ones(n)
    yval_ = np.ones(n)
    
    y_[y == 0] = -1
    yval_[yval == 0] = -1
    
    
    loss = []
    acc = []
    
    loss_val = []
    acc_val = []
    
    
    alphas = []
    
    for l in lambd :
        cnt = 0
        
        P_t, W_t = np.eye(n), np.eye(n)
        z_t = K@ np.ones(n) - y_
        alpha_t = np.ones(n)
        diff_alpha = np.inf


        while (diff_alpha > tresh) and (cnt < maxIter):

            old_alpha = alpha_t
            
            ## Solving dual using CVXOpt
            #P = matrix(2*((K @ W_t @ K)/n + l*K))
            #q = matrix((-2*K@W_t@y_)/n)
            #solvers.options['show_progress'] = False
            #sol=solvers.qp(P, q)
            #alpha_t = sol['x']
            #alpha_t = np.reshape(alpha_t,-1)  
            
            alpha_t = solveWKRR(K, W_t, z_t, y_, l)

            m_t = K@alpha_t
            sigma_m = sigmoid(m_t)
            sigma_my = sigmoid(-y_*m_t)

            P_t = - np.diag(sigma_my)
            W_t = np.diag(sigma_m * (1-sigma_m))

            z_t = m_t - (P_t@y_)/(sigma_m * (1-sigma_m))

            diff_alpha = np.linalg.norm(alpha_t - old_alpha)
            cnt+=1
            if cnt % 10 == 0:
                print(l, cnt)
        
        loss_lambda = logistic_loss(y_, K@alpha_t)
        acc_lambda = accuracy(y,K@alpha_t, mode="SVM")
        
        loss_lambdaval = logistic_loss(yval_, Kval@alpha_t)
        acc_lambdaval = accuracy(yval,Kval@alpha_t, mode="SVM")

        
        print(f"***********lambda = {l}***********")
        print(f"Training: loss = {loss_lambda:.4f}, accuracy = {acc_lambda:.6f}")
        print(f"Validation: loss = {loss_lambdaval:.4f}, accuracy = {acc_lambdaval:.6f}")
        
        
        loss += [loss_lambda]
        acc += [acc_lambda]
        
        loss_val += [loss_lambdaval]
        acc_val += [acc_lambdaval]
        
        alphas +=[alpha_t]
        
    return(alphas, loss, acc, loss_val, acc_val)

def hinge_loss(y_true, y_pred):
    n = y_true.shape[0]
    term = np.maximum(1-y_true*y_pred, 0)
    return(np.sum(term)/n)

def SVM(K, y, K_val, y_val, lambd):
    """
    
    """
    assert K.shape[0] == y.shape[0]
    n = K.shape[0]
    
    y_ = np.ones(n)
    yval_ = np.ones(n)
    
    y_[y == 0] = -1
    yval_[y_val == 0] = -1
    
    y_preds, y_preds_val = [], []
    losses, losses_val = [], []
    accuracies, accuracies_val = [], []
    alphas = []
    
    
    for l in lambd :

        ## Solving dual using CVXOpt
        P = matrix(K)
        q = matrix(-y_)
        D = np.diag(-y_)
        G = matrix(np.vstack((D,-D)))
        h = matrix(np.concatenate((np.zeros(n), 1/(2*l*n) * np.ones(n)), axis=0))
        solvers.options['show_progress'] = False
        sol=solvers.qp(P, q, G, h)
        alpha = sol['x']
        alpha = np.reshape(alpha,-1)               

        ## predictions
        # training
        pred_l = K @ alpha
        y_preds += [pred_l]
        loss_l = hinge_loss(y_, pred_l)
        acc_l = accuracy(y, pred_l, mode="SVM")

        
        # validation
        pred_l_val = K_val@alpha
        y_preds_val += [pred_l_val]
        loss_l_val = hinge_loss(yval_, pred_l_val)
        acc_l_val = accuracy(y_val,pred_l_val, mode="SVM")
        

        print(15*"-", f" lambda = {l} ", 15*"-")
        print(f"Training: loss = {loss_l:.6f}, accuracy = {acc_l:.6f}")
        print(f"Validation: loss = {loss_l_val:.6f}, accuracy = {acc_l_val:.6f}")
        
        losses += [loss_l]
        accuracies += [acc_l]
        
        losses_val += [loss_l_val]
        accuracies_val += [acc_l_val]
    
        alphas +=[alpha] 
        
    return(alphas, losses, accuracies, losses_val, accuracies_val)
            









