import numpy as np

def window_per_slice(slice_data): # 2D data [ [x1, y1], [x2, y2], [...] ]
    reshaped_points = np.array(slice_data).reshape(-1, 3)
    x, y = reshaped_points[:, 0], reshaped_points[:, 1]
    x_max, x_min = np.max(np.array(x)), np.min(np.array(x))
    y_max, y_min = np.max(np.array(y)), np.min(np.array(y))
    return x_max-x_min, y_max-y_min

def total_window_list(ROI_contours):
    slice_size = len(ROI_contours)
    wx_list, wy_list = [], []
    for slice in range(slice_size):
        curr_slice = ROI_contours[slice]
        curr_wx, curr_wy = window_per_slice(curr_slice)
        wx_list.append(curr_wx)
        wy_list.append(curr_wy)
    return wx_list, wy_list

def window_size(wx_list, wy_list, spacing):
    wx_max, wx_min = np.max(np.array(wx_list)), np.min(np.array(wx_list))
    wy_max, wy_min = np.max(np.array(wy_list)), np.min(np.array(wy_list))
    return int(wx_max-wx_min)/spacing, int(wy_max-wy_min)/spacing

def per_slice_mapping(slice_data, w_size, center, spacing):
    reshaped_points = np.array(slice_data).reshape(-1, 3)
    x, y = reshaped_points[:, 0], reshaped_points[:, 1]

    image = np.zeros(w_size)
    for i in range(len(x)):
        x_pos, y_pos = int( (x[i] - center[0]) / spacing ), int( (y[i] - center[1]) / spacing )
        image[x_pos, y_pos] = 1
    return image

"""
def rasterize_contour(contour_points):
    from skimage.draw import polygon
    # Rasterize contour into a binary mask.
    reformed_points = np.array(contour_points).reshape(-1, 3)
    x, y = reformed_points[:, 0], reformed_points[:, 1]
    
    # Normalize to grid
    x = ((x - x.min()) / 0.3).astype(int)
    y = ((y - y.min()) / 0.5).astype(int)
    
    # Create binary mask
    mask = np.zeros((300, 400), dtype=bool)
    rr, cc = polygon(y, x, shape=(300, 400))
    mask[rr, cc] = True
    return mask

def concat_3d(contours, ROI_num):
    curr_roi_dict = contours[ROI_num]
    slice_size = len(curr_roi_dict.keys())
    final_mask = np.zeros((300, 400, slice_size))
    for slice_num in list(curr_roi_dict.keys()):
        curr_slice_data = curr_roi_dict[slice_num]
        curr_mask = rasterize_contour(curr_slice_data)
        final_mask[:, :, slice_num] = curr_mask
    return final_mask
"""