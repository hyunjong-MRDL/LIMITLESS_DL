import data, utils, processing, metrics

root = "D:/Datasets/CT_Recon/Breast/"

exit_flag = 1
while True:
    exit_flag = int(input("If you want to terminate this program, please enter 0, or else enter 1: "))
    if exit_flag == 0: break
    print()
    print("Reference data\n")
    recon_1 = input("Desired Reconstruction Method [Type 'Aice' or 'AIDR']: ")
    seg_1 = input("Desired Segmentation Method: [Type 'Breast' or 'Onco']: ")
    print()
    print("Comparative data\n")
    recon_2 = input("Desired Reconstruction Method [Type 'Aice' or 'AIDR']: ")
    seg_2 = input("Desired Segmentation Method: [Type 'Breast' or 'Onco']: ")
    print()

    path_1 = data.get_RT_path(root, 0, recon_1, seg_1)  # ID1_Aice_manual
    path_2 = data.get_RT_path(root, 0, recon_2, seg_2)  # ID1_AIDR_manual

    structures_1 = data.get_ROI_structures(path_1)
    contours_1 = data.get_ROI_contours(path_1)

    structures_2 = data.get_ROI_structures(path_2)
    contours_2 = data.get_ROI_contours(path_2)

    total_contours_1 = data.get_contour_data(contours_1)
    total_contours_2 = data.get_contour_data(contours_2)

    roi_names_1 = utils.get_ROI_names(path_1)
    roi_names_2 = utils.get_ROI_names(path_2)

    matched_ROIs = utils.match_ROIs(roi_names_1, roi_names_2)
    utils.print_ROI_names(matched_ROIs)

    ROI_num = int(input("Interested ROI number: "))
    print()

    mask_1 = processing.concat_3d(total_contours_1, ROI_num)
    mask_2 = processing.concat_3d(total_contours_2, ROI_num)

    result = metrics.dice_coefficient(mask_1, mask_2)
    print(result)
    print()

print("Program terminated.")