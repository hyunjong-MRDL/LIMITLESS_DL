import numpy as np

"""Cut Data Amplitudes by Beam-Enabled Moments"""
def cut_by_beams(data_Times, data_Amps, beam_Times):
    dt = 0.015
    cutted_Amps = np.zeros(len(data_Amps))
    for i in range(len(beam_Times)//2):
        for j in range(len(data_Times)):
            if (data_Times[j] > beam_Times[2*i]):
                interval = int( (beam_Times[2*i+1] - beam_Times[2*i]) / dt )
                cutted_Amps[j:j+interval] = data_Amps[j:j+interval]
                break
    return cutted_Amps

# """Cut Data Amplitudes by Beam-Enabled Moments"""
# def cut_2(data_Times, data_Amps, beam_Times):
#     cutted_Amps = np.zeros(len(data_Amps))
#     i = 0
#     for j in range(len(data_Times)-1):
#         if (i >= len(beam_Times)): break
#         if (data_Times[j] > beam_Times[i]):
#             cutted_Amps[j] = data_Amps[j]
#         elif (data_Times[j] > beam_Times[i+1]):
#             if (data_Times[j+1] > beam_Times[i+2]): i+= 2
#             else: cutted_Amps[j] = 0
#     return cutted_Amps

"""Sectionize beam-enabled intervals"""
"""There are more than one beam session in even one field"""
def beam_enabling_intervals(data_Times, data_Amps, beam_Times):
    dt = 0.015
    total_intervals = []
    for i in range(len(beam_Times)//2):
        for j in range(len(data_Times)):
            if (data_Times[j] > beam_Times[2*i]):
                interval = int( (beam_Times[2*i+1] - beam_Times[2*i]) / dt )
                total_intervals.append(data_Amps[j:j+interval])
                break
    return total_intervals, len(total_intervals)

"""Best-fitting line of given data points"""
def regression_line(intv_Amps):
    dt = 0.015
    Times = [t*dt for t in range(len(intv_Amps))]
    slope, intercept = np.polyfit(Times, intv_Amps, deg=1)
    fitted_line = [slope*t+intercept for t in Times]
    return fitted_line

def dilate_metrics(data_Times, beam_Times, avg_lvls, fitted_lines):
    dilated_avgs = np.zeros(len(data_Times))
    dilated_line = np.zeros(len(data_Times))

    for i in range(len(beam_Times)//2):
        curr_avg = avg_lvls[i]
        curr_line = fitted_lines[i]
        for j in range(len(data_Times)):
            if (data_Times[j] > beam_Times[2*i]):
                dilated_avgs[j:j+len(curr_line)] = curr_avg * np.ones(len(curr_line))
                dilated_line[j:j+len(curr_line)] = curr_line
                break

    return dilated_avgs, dilated_line