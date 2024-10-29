import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from Dataset import test_ctrl_path, test_ASD_path

def plot(target, pred):
    auc = roc_auc_score(target, pred)

    fpr, tpr, thresholds = roc_curve(target, pred)
    J=tpr-fpr
    idx = np.argmax(J)

    best_thresh = thresholds[idx]
    sens, spec = tpr[idx], 1-fpr[idx]

    test_patient = len(test_ASD_path)
    test_normal = len(test_ctrl_path)

    acc = (sens*test_patient + spec * test_normal) / (test_patient + test_normal)
    auc = roc_auc_score(target, pred)

    plt.title("Roc Curve")
    plt.plot([0,1], [0,1], linestyle='--', markersize=0.01, color='black')
    plt.plot(fpr, tpr, marker='.', color='black', markersize=0.05)
    plt.scatter(fpr[idx], tpr[idx], marker='o', s=200, color='r',
                label = 'Sensitivity : %.3f (%d / %d), \nSpecificity = %.3f (%d / %d), \nAUC = %.3f , \nACC = %.3f (%d / %d)' % (sens, (sens*test_patient), test_patient, spec, (spec*test_normal), test_normal, auc, acc, sens*test_patient+spec*test_normal, test_patient+test_normal))
    plt.legend()
    plt.savefig("./Figures/Test.png")