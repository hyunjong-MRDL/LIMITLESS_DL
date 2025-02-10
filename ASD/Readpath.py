import os
import numpy as np

EXCEPTION = ["536", "537", "544", "955", "716", "733", "1040", "1727", "1734", "1746", "2902", "2909", "3278", "3395", "3466", "3564", "3603", "3624", "3635", "3663", "4043", "4044", "4047", "4067",
             "4082", "4088", "4586", "4740", "4808", "s4-13", "s4-29", "s4-31", "s4-73", "s4-97", "s4-98", "s4-102", "s4-107", "s4-111", "s4-174", "s4-234", "s4-238", "s4-240", "s4-266", "s4-282"]

root = "D:/Datasets/ASD/"

def read_path(path):  # oirg.mgz (freesurfer images, T1 ONLY)
    path = path + "DATA/"
    ctrl_path, ASD_path = [], []
    ctrl_label, ASD_label = [], []

    for diagnosis in os.listdir(path):
        diag_path = path + diagnosis + "/"

        for ID in os.listdir(diag_path):
            if ID in EXCEPTION:
                continue
            im_path = f"{diag_path}{ID}/freesurfer/mri/brain.mgz"
            if diagnosis == "ASD":
                ASD_path.append(im_path)
                ASD_label.append(np.array([1, 0]))
            else:
                ctrl_path.append(im_path)
                ctrl_label.append(np.array([0, 1]))
    
    return ctrl_path, ctrl_label, ASD_path, ASD_label

ctrl_path, ctrl_label, ASD_path, ASD_label = read_path(root)    