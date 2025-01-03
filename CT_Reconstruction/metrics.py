import numpy as np

def dice_coefficient(array1, array2):
    intersection = np.sum(array1 * array2)
    sum_arrays = np.sum(array1) + np.sum(array2)
    
    # Avoid division by zero
    if sum_arrays == 0:
        return 1.0 if np.array_equal(array1, array2) else 0.0

    # Dice coefficient calculation
    dice = 2 * intersection / sum_arrays
    return dice