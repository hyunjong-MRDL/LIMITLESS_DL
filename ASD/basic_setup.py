import os
import torch
import random
import numpy as np
import nibabel as nib

device = 'cuda' if torch.cuda.is_available() else 'cpu'

orientation_list = ["sag", "axi", "cor"]
#### SAGITTAL : 옆에서 촬영한 영상 ####
#### AXIAL : 위에서 촬영한 영상 ####
#### CORONAL : 앞에서 촬영한 영상 ####
train_orientation = 0  # ["sagittal", "axial", "coronal"] -> index 입력

hyperparameters = {"SEED": 42,
                   "TYPE": 3,  # 2: 2D image, 3: 3D voxel
                   "TEST_PORTION": 0.2,
                   "EPOCHS": 10,
                   "BATCH_SIZE": 1,
                   "LR": 1e-4}

image_dimension = hyperparameters["TYPE"]
if image_dimension == 3:
    model_name = "Res3d"
    model_save_path = f"E:/LIMITLESS_DL/ASD/save_path/{model_name}(CE_Loss).pt"
elif image_dimension == 2:
    model_name = "Res2d"
    model_save_path = f"E:/LIMITLESS_DL/ASD/save_path/{model_name}_epo10(CE_Loss).pt"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True