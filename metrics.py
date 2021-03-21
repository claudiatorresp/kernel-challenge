import numpy as np
import pandas as pd

def write_predictions_csv(test_kernels, test_alphas, path, mode="SVM"):
    
    n = 1000
    predictions = np.zeros(3000, dtype=int)
    
    for i in range(3):
        y_pred = test_kernels[i] @ test_alphas[i]
        if mode == 'y_pred':
            y_pred_ = np.ones(n)
            y_pred_[y_pred < 0] = 0
        else:
            y_pred_ = np.zeros(n)
            y_pred_[y_pred >= 0.5] = 1
   
        predictions[1000*i:1000*(i+1)] = y_pred_
    
    #predictions = predictions.astype(int)
    pred = pd.DataFrame({"Bound" : predictions})
    print("saving predictions")
    pred.to_csv(path, index=True,index_label="Id")
    #np.savetxt("data/Ytest_KRR.csv", predictions, header = "Id, Bound", delimiter =",")
    print("saved predictions")
        

