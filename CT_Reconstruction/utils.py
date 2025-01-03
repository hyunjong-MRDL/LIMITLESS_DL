import numpy as np
import matplotlib.pyplot as plt
import data

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

def plot_coordinates(slice_contours):
    for ROI_num in list(slice_contours.keys()):
        curr_roi_dict = slice_contours[ROI_num]
        for slice_num in list(curr_roi_dict.keys()):
            curr_slice_data = np.array(curr_roi_dict[slice_num]).reshape(-1, 3)
            x, y = curr_slice_data[:, 0], curr_slice_data[:, 1]
            plt.figure(), plt.plot(x, y)
            plt.savefig(f'./Figures/A_IR_manual/ROI{ROI_num}_slice{slice_num}.jpg'), plt.close()
    return