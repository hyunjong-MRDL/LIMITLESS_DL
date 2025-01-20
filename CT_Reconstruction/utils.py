import os, data
import numpy as np
import matplotlib.pyplot as plt

total_ROI_names = ["External", "Bowels", "Lungs", "Breast_L", "Breast_R", "Spleen", "Liver", "Kidney_L", "Kidney_R",
                   "Gallbladder", "Heart", "Cavity_Oral", "Brachi_L", "Brachi_R", "Sigmoid_Colon", "Bowel_Large",
                   "Bowel_Samll", "Lung_R", "Trachea", "Esophagus", "Stomach", "Duodenum", "Glnd_Thyroid", "CaudaEquina",
                   "Parotid_L", "Parotid_R", "Glnd_Submand_L", "Glnd_Submand_R"]

## 비슷한 이름 처리하는 알고리즘

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    return

def get_ROI_names(RT_path):
    ROI_namedict = dict()
    RT_structuresets = data.get_ROI_structures(RT_path)
    RT_size = len(RT_structuresets)
    for i in range(RT_size):
        curr_num = RT_structuresets[i].ROINumber
        curr_name = RT_structuresets[i].ROIName
        ROI_namedict[curr_name] = curr_num
    return ROI_namedict

def print_ROI_names(ROI_namedict):
    for ROI_name in ROI_namedict:
        print(f"{ROI_namedict[ROI_name]}: {ROI_name}")
    print()
    return

def match_ROIs(ROI_namedict_1, ROI_namedict_2):
    matched_names = []
    ROI_names_1 = list(ROI_namedict_1.keys())
    ROI_names_2 = list(ROI_namedict_2.keys())
    for roi in total_ROI_names:
        if (roi in ROI_names_1) and (roi in ROI_names_2):
            matched_names.append(roi)
    return matched_names

def plot_coordinates(ID, recon, seg, slice_contours):
    plot_folder = f"./Figures/{ID}/{recon}_{seg}/"
    createFolder(plot_folder)

    for ROI_num in list(slice_contours.keys()):
        curr_roi_dict = slice_contours[ROI_num]
        for slice_num in list(curr_roi_dict.keys()):
            curr_slice_data = np.array(curr_roi_dict[slice_num]).reshape(-1, 3)
            x, y = curr_slice_data[:, 0], curr_slice_data[:, 1]
            filename = f"{plot_folder}ROI{ROI_num}_slice{slice_num}.jpg"
            if os.path.exists(filename):
                print(f"Contour plot [{filename}] already exists.")
            else:
                plt.figure(), plt.plot(x, y)
                plt.savefig(filename), plt.close()
                print(f"Contour plot [{filename}] has saved successfully.")
    return