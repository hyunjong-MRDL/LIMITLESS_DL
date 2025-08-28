from utils import construct_total_dict, createFolder
from processing import merge_data
from plot import plot_whole_data

if __name__ == "__main__":

    root = './DATA/'
    breathhold_dict, full_dict = construct_total_dict(root)
    createFolder(f"./Plots/Breathhold")
    createFolder(f"./Plots/Full")

    scan_list = list(breathhold_dict.keys())
    for scan_num in scan_list:
        b_Time, b_Amp, b_Beams, b_States = merge_data(breathhold_dict, scan_num)
        f_Time, f_Amp, f_Beams, f_States = merge_data(full_dict, scan_num)

    plot_whole_data("Breathhold", b_Time, b_Amp)
    plot_whole_data("Full", f_Time, f_Amp)