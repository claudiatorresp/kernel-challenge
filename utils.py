from csv import reader
import numpy as np
import pandas as pd


def features_into_array(path):
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        X = list()
        if header != None:
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                X.append(np.array(row[1]))
                
    X = np.array(X) ## dtype might be changed in something more convenient. For now, dtype = "<U1"
    return X
    
def write_predictions_csv(y_tests, path, mode="SVM"):
    """
    Takes as inputs a list of the predictions [K0@alpha0, K1@akpha1, K2@alpha2] and saves them to the given path
    If mode == "SVM" means the inputs are in [-1,1]
    Otherwise, the inputs are supposed to be in [0,1]
    """
    
    n = 1000
    predictions = np.zeros(3*n, dtype=int)
    
    for i in range(3):
        y_pred = y_tests[i]
        if mode == 'SVM':
            y_pred_ = np.ones(n)
            y_pred_[y_pred < 0] = 0
        else:
            y_pred_ = np.zeros(n)
            y_pred_[y_pred >= 0.5] = 1
   
        predictions[n*i:n*(i+1)] = y_pred_
    
    pred = pd.DataFrame({"Bound" : predictions})
    print("saving predictions to " + path)
    pred.to_csv(path, index=True,index_label="Id")
    print("saved predictions")
    return(predictions)