from classifiers import *
from metrics import *
from mismatch_spectrum_kernel import *
from weighted_majority_voting import*
from utils import *
import time

## Load the DATA
print("loading data")
Xtr0 = features_into_array("data/Xtr0.csv")
Xte0 = features_into_array("data/Xte0.csv")
Ytr0 = np.genfromtxt("data/Ytr0.csv", delimiter=',', skip_header=1)

Xtr1 = features_into_array("data/Xtr1.csv")
Xte1 = features_into_array("data/Xte1.csv")
Ytr1 = np.genfromtxt("data/Ytr1.csv", delimiter=',', skip_header=1)

Xtr2 = features_into_array("data/Xtr2.csv")
Xte2 = features_into_array("data/Xte2.csv")
Ytr2 = np.genfromtxt("data/Ytr2.csv", delimiter=',', skip_header=1)


#train_test_split
train_idx_0 = np.load("train_test_split/train_idx_0.npy").astype(int)
train_idx_1 = np.load("train_test_split/train_idx_1.npy").astype(int)
train_idx_2 = np.load("train_test_split/train_idx_2.npy").astype(int)

val_idx_0 = np.load("train_test_split/val_idx_0.npy").astype(int)
val_idx_1 = np.load("train_test_split/val_idx_1.npy").astype(int)
val_idx_2 = np.load("train_test_split/val_idx_2.npy").astype(int)

Xtr0_ = Xtr0[train_idx_0]
Xtr1_ = Xtr1[train_idx_1 - 2000]
Xtr2_ = Xtr2[train_idx_2 - 4000]

ytr0 = Ytr0[train_idx_0]
ytr1 = Ytr1[train_idx_1 - 2000]
ytr2 = Ytr2[train_idx_2 - 4000]


Xval0_ = Xtr0[val_idx_0]
Xval1_ = Xtr1[val_idx_1 - 2000]
Xval2_ = Xtr2[val_idx_2 - 4000]

yval0 = Ytr0[val_idx_0]
yval1 = Ytr1[val_idx_1 - 2000]
yval2 = Ytr2[val_idx_2 - 4000]
print("finished loading data")
print("")
print("Ready to compute kernels")

Kernels_tr_0 = []
Kernels_val_0 = []
Kernels_te_0 = []

Kernels_tr_1 = []
Kernels_val_1 = []
Kernels_te_1 = []

Kernels_tr_2 = []
Kernels_val_2 = []
Kernels_te_2 = []
m = 1

for k in range(5,11):
    print("*"*15 + "treating k = " + str(k) + 15*"*")
    print("."*15 + "treating m = " + str(1) + 15*".")
    start_time = time.time()
    alphabet_km = all_possible_substrings_mismatch2(k,m)
    print("--- Found alphabet in %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    K_tr0, K_val0, K_te0 = mismatch_spectrum_kernel2(Xtr0_, Xval0_, Xte0, k=k, m=m, alphabet=alphabet_km)
    Kernels_tr_0 += [K_tr0]
    Kernels_val_0 += [K_val0]
    Kernels_te_0 += [K_te0]

    K_tr1, K_val1, K_te1 = mismatch_spectrum_kernel2(Xtr1_, Xval1_, Xte1, k=k, m=m, alphabet=alphabet_km)
    Kernels_tr_1 += [K_tr1]
    Kernels_val_1 += [K_val1]
    Kernels_te_1 += [K_te1]

    K_tr2, K_val2, K_te2 = mismatch_spectrum_kernel2(Xtr2_, Xval2_, Xte2, k=k, m=m, alphabet=alphabet_km)
    Kernels_tr_2 += [K_tr2]
    Kernels_val_2 += [K_val2]
    Kernels_te_2 += [K_te2]

    print("--- Computed all the kernels in %s seconds ---" % (time.time() - start_time))
    print("")
    print("")
    print("")

    
print("Ready to find SVMs alphas with given regularizations")

C_mismatch_0 = [0.0004, 0.0002, 0.0001, 0.0004, 0.0003, 0.0006]
C_mismatch_1 = [0.0003, 0.0004, 0.0002, 0.0005, 0.0004, 0.0003]
C_mismatch_2 = [0.0002, 0.0005, 0.0005, 0.0004, 0.0004, 0.001]

alphas_0 = []
alphas_1 = []
alphas_2 = []


for i in range(6):
    print("*"*15 + f"Treating mismatch kernels for k = {i+5} " + "*"*15)
    print("")
    print("************* SVM for dataset 0*************\n")
    
    alphas_tr0_mismatch = SVM(Kernels_tr_0[i], ytr0[:,1], Kernels_val_0[i],
                                       yval0[:,1], [C_mismatch_0[i]])
    
    print("")
    print("************* SVM for dataset 1 *************\n")
    
    alphas_tr1_mismatch = SVM(Kernels_tr_1[i], ytr1[:,1], Kernels_val_1[i],
                                       yval1[:,1], [C_mismatch_1[i]])
    print("")
    print("************* SVM for dataset 2 *************\n")
    alphas_tr2_mismatch = SVM(Kernels_tr_2[i], ytr2[:,1], Kernels_val_2[i],
                                       yval2[:,1], [C_mismatch_2[i]])
    print("")
    print("")
    
    alphas_0 += alphas_tr0_mismatch
    alphas_1 += alphas_tr1_mismatch
    alphas_2 += alphas_tr2_mismatch
    
print("Ready to predict - Majority Voting")
print("")
print("Dataset 0")

y_te0_pred_mv = no_weighted_mv(Kernels_tr_0,
            Kernels_val_0,
            alphas_0, ytr0[:,1], yval0[:,1],
            K_test =Kernels_te_0)
print("")
print("Dataset 1")
y_te1_pred_mv = no_weighted_mv(Kernels_tr_1,
            Kernels_val_1,
            alphas_1, ytr1[:,1], yval1[:,1],
            K_test =Kernels_te_1)
print("")
print("Dataset 2")
y_te2_pred_mv = no_weighted_mv(Kernels_tr_2[1:],
            Kernels_val_2[1:],
            alphas_2[1:], ytr2[:,1], yval2[:,1],
            K_test =Kernels_te_2[1:])
print("")

write_predictions_csv([y_te0_pred_mv, y_te1_pred_mv, y_te2_pred_mv], "predictions/y_tes_mv.csv", mode="SVM")


print("Ready to predict - Weighted Majority Voting")
print("")
print("Dataset 0")
y_te0_pred_wmv = weighted_mv(Kernels_tr_0,
            Kernels_val_0,
            alphas_0, ytr0[:,1], yval0[:,1],
            K_test =Kernels_te_0)
print("")
print("Dataset 1")
y_te1_pred_wmv = weighted_mv(Kernels_tr_1,
            Kernels_val_1,
            alphas_1, ytr1[:,1], yval1[:,1],
            K_test =Kernels_te_1)
print("")
print("Dataset 2")
y_te2_pred_wmv = weighted_mv(Kernels_tr_2[1:],
            Kernels_val_2[1:],
            alphas_2[1:], ytr2[:,1], yval2[:,1],
            K_test =Kernels_te_2[1:])
print("")
write_predictions_csv([y_te0_pred_wmv, y_te1_pred_wmv, y_te2_pred_wmv], "predictions/y_tes_wmv.csv", mode="SVM")

