import os
from pydicom import dcmread

"""Folder names are IDs"""
def get_IDs(path):
    return os.listdir(path)

"""AiCE or AIDR"""
def recon_path(path, method):
    return f"{path}{method}/"

""""Breast(manual) or Onco(auto)"""
def seg_path(path, method):
    for p in os.listdir(path):
        if method in p: x = p
    return f"{path}{x}/"

"""Filenames are in the order of CT0, CT1, ..., RT"""
"""            Recon: 'Aice' or 'AIDR'            """
"""       Segmentation: 'Breast' or 'Onco'        """
def get_RT_path(path, ID_idx, recon, seg):
    path = f"{path}{get_IDs(path)[ID_idx]}/"
    path = seg_path(recon_path(path, recon), seg)
    return f"{path}{os.listdir(path)[-1]}"

def get_ROI_structures(RT_path):
    ROI_structures = dcmread(RT_path).StructureSetROISequence
    return ROI_structures

def get_ROI_contours(RT_path):
    ROI_contours = dcmread(RT_path).ROIContourSequence
    return ROI_contours

def get_contour_data(ROI_contours):
    total_contours = dict()
    for roi_index in range(len(ROI_contours)):
        curr_roi_seq = ROI_contours[roi_index]
        curr_contour_dict = dict()
        if hasattr(curr_roi_seq, "ContourSequence") and curr_roi_seq.ContourSequence:
            for slice_num in range(len(curr_roi_seq.ContourSequence)):
                curr_contour_dict[slice_num] = curr_roi_seq.ContourSequence[slice_num].ContourData
        # When printing this message, make sure to include which DATA is being used (Recon/Segmentation methods)
        else: print(f"WARNING: ROI index ({roi_index}) has no Contour Sequence.\n")
        total_contours[curr_roi_seq.ReferencedROINumber] = curr_contour_dict
    return total_contours