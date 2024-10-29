import os
import numpy as np

def read_path(root, type):
    if type == 0:  # Training set
        path = f"{root}train/"
    elif type == 1:  # Testing set
        path = f"{root}test/"
    else:  # Validation set
        path = f"{root}val/"
    
    n_path, n_label = [], []
    v_path, v_label = [], []
    b_path, b_label = [], []

    for diagnosis in os.listdir(path):
        diag_path = f"{path}{diagnosis}/"
        for img in os.listdir(diag_path):
            if diagnosis == "NORMAL":
                n_path.append(f"{diag_path}{img}")
                n_label.append(np.array([0,0,1]))
            elif diagnosis == "VIRUS":
                v_path.append(f"{diag_path}{img}")
                v_label.append(np.array([0,1,0]))
            else:
                b_path.append(f"{diag_path}{img}")
                b_label.append(np.array([1,0,0]))
    
    total_path = n_path + v_path + b_path
    total_label = n_label + v_label + b_label
    
    return total_path, total_label