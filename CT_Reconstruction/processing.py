import numpy as np

def window_in_slice(slice_data):
    return

def rasterize_contour(contour_points):
    from skimage.draw import polygon
    """Rasterize contour into a binary mask."""
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