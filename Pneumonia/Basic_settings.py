import os
import torch
import random
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

hyperparameters = {"SEED": 42,
                   "IMSIZE": 224,
                   "EPOCHS": 15,
                   "BATCH": 64,
                   "LR": 1e-4,
                   "model_save_path": "C:/Users/PC00/Desktop/HJ/AICOSS_2023(Pneumonia)/model_save_path/practice.pt"}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True