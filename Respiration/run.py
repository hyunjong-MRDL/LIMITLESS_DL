from utils import construct_total_dict, createFolder
from processing import merge_data, split_beams
from plot import plot_by_index
from metric import average_level, calc_reprod, maximum_change, calc_stab

if __name__ == "__main__":

    root = './DATA/'
    breathhold_dict, full_dict = construct_total_dict(root)
    createFolder(f"./Plots/Breathhold")
    createFolder(f"./Plots/Full")

    scan_list = list(breathhold_dict.keys())
    for scan_num in scan_list:
        b_Time, b_Amp, b_Beams, b_States = merge_data(breathhold_dict, scan_num)
        f_Time, f_Amp, f_Beams, f_States = merge_data(full_dict, scan_num)
        b_timedict, b_beamdict = split_beams(b_Time, b_Amp, b_Beams, b_States)
        f_timedict, f_beamdict = split_beams(f_Time, f_Amp, f_Beams, f_States)

    b_beam_index = list(b_beamdict.keys())
    f_beam_index = list(f_beamdict.keys())
    b_avg_levels, f_avg_levels = [], []
    b_dMaxs, f_dMaxs = [], []
    for index in b_beam_index:
        plot_by_index("Breathhold", b_timedict, b_beamdict, index)
        plot_by_index("Full", f_timedict, f_beamdict, index)
        # reproducibility
        curr_b_avg_lv = average_level(b_beamdict[index])
        b_avg_levels.append(curr_b_avg_lv)
        curr_f_avg_lv = average_level(f_beamdict[index])
        f_avg_levels.append(curr_f_avg_lv)
        # stability
        curr_b_dMax = maximum_change(b_timedict[index], b_beamdict[index])
        b_dMaxs.append(curr_b_dMax)
        curr_f_dMax = maximum_change(f_timedict[index], f_beamdict[index])
        f_dMaxs.append(curr_f_dMax)

    b_reprod, f_reprod = calc_reprod(b_avg_levels), calc_reprod(f_avg_levels)
    b_stab, f_stab = calc_stab(b_dMaxs), calc_stab(f_dMaxs)

    print(f"\n\n\t\t\tReproducibility\t\t\nBreathhold: {b_reprod} Full: {f_reprod}\n\n")
    print(f"\t\t\tStability\t\t\nBreathhold: {b_stab} Full: {f_stab}")