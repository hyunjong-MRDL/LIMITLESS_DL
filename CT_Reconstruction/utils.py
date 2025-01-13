import os, data
import numpy as np
import matplotlib.pyplot as plt

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    return

def get_ROI_names(RT_path):
    ROI_names = dict()
    RT_structuresets = data.get_ROI_structures(RT_path)
    RT_size = len(RT_structuresets)
    for i in range(RT_size):
        curr_num = RT_structuresets[i].ROINumber
        curr_name = RT_structuresets[i].ROIName
        ROI_names[curr_num] = curr_name
    return ROI_names

def print_ROI_names(ROI_name_dict):
    ROI_nums = list(ROI_name_dict.keys())
    print(f"The common ROIs shared by two data are shown below:")
    for num in ROI_nums:
        print(f"{num}: {ROI_name_dict[num]}")
    print()
    return

def match_ROIs(ROI_names_1, ROI_names_2):
    ROI_nums_1 = list(ROI_names_1.keys())
    ROI_nums_2 = list(ROI_names_2.keys())
    matched_nums = list(set(ROI_nums_1) & set(ROI_nums_2))
    return {key: ROI_names_1[key] for key in matched_nums}

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