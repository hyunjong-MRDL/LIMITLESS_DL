import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""호흡 데이터 분석에 필요한 모든 함수 구현"""

def patient_listing(root):
    """Data organization (Listing ALL patients in a dictionary)"""
    ### The subroot directories describe either the data type (STATIC or ARC) or the date of acquisition
    # Inside the subroot directory -> There are THREE types of subdirectories
    # 1. Subdirectory has ONE folder of total data from a single patient (MOST SIMPLE CASE)
    # 2. Subdirectory has folders of multiple patients
    # 3. Subdirectory has TWO different types of data - Trained / Untrained set of patients
    patients_to_analyze = dict()
    for datatype in os.listdir(root):
        subroot = os.path.join(root, datatype)
        subdirs = [os.path.join(subroot, directory) for directory in os.listdir(subroot) if os.path.isdir(f"{subroot}/{directory}")]
        if len(subdirs) == 1:
            """Case 1: Most simple case"""
            patients_to_analyze[datatype] = subdirs
        else:
            if not any("education" in directory for directory in subdirs):
                """Case 2: Multiple patients"""
                IDs = [os.path.join(subroot, directory) for directory in os.listdir(subroot) if os.path.isdir(f"{subroot}/{directory}")]
                patients_to_analyze[datatype] = IDs
            else:
                """Case 3: Training information included"""
                trained = os.path.join(subroot, "education")
                untrained = os.path.join(subroot, "non-education")
                trained_IDs = [os.path.join(trained, directory) for directory in os.listdir(trained) if os.path.isdir(f"{trained}/{directory}")]
                untrained_IDs = [os.path.join(untrained, directory) for directory in os.listdir(untrained) if os.path.isdir(f"{untrained}/{directory}")]
                patients_to_analyze[f"{datatype}_trained"] = trained_IDs
                patients_to_analyze[f"{datatype}_untrained"] = untrained_IDs
    return patients_to_analyze

def find_continuous_fields(total_list):
    field_duplicates = []
    for pt_path in total_list:
        ID = os.path.normpath(pt_path).split(os.sep)[-1]
        for fraction in sorted(os.listdir(pt_path), key=int):
            fx_path = os.path.join(pt_path, fraction)
            fields_in_fx = []
            for field_path in sorted(os.listdir(fx_path)):
                field_data = os.path.join(fx_path, field_path)
                match = re.search(r'field(\d+)', field_data, re.IGNORECASE)
                if match is not None:
                    curr_field = match.group(1)
                    if curr_field not in fields_in_fx:
                        fields_in_fx.append(curr_field)
                    else:
                        field_duplicates.append([ID, fraction, curr_field])
    return field_duplicates

def read_field_data(field_path):
    """How to read a field-data (all data are field-data)"""
    """Multiple fields are applied to the patients in each fraction"""
    data_Times, data_Amps = [], []
    beam_Times, beam_States = [], []
    THICK_cnt, thin_cnt = 0, 0
    data_flag, beam_flag  = False, False

    with open(field_path, "r") as file: # Open file
        for line in file:
            if "=============" in line:
                THICK_cnt += 1
            if "-------------" in line:
                thin_cnt += 1
            
            ### Read time-data
            if (THICK_cnt >= 4) and (thin_cnt == 1):
                if data_flag and line == "\n": data_flag = False
                if data_flag:
                    time, amplitude = line.strip().split("\t")
                    data_Times.append(float(time))
                    data_Amps.append(float(amplitude) * 10.0) # cm to mm
                if (THICK_cnt >= 4 and thin_cnt <= 1) and ("Amplitude" in line): data_flag = True
            
            ### Read beam-data
            if (THICK_cnt >= 6):
                if beam_flag and line == "\n": break
                elif beam_flag:
                    time, state = line.strip().split("\t")
                    beam_Times.append(float(time))
                    beam_States.append(int(state))
                elif ("Time" in line): beam_flag = True

    return (data_Times, data_Amps), (beam_Times, beam_States)

def beam_modification(beam_Times):
    modified_beam_Times = []
    i = 0
    if len(beam_Times) <= 2: return beam_Times  # Only ONE beam-session
    else:
        modified_beam_Times.append(beam_Times[0])  # Fill in 1st ON-time
        while i < (len(beam_Times)//2-1):
            if (beam_Times[2*i+2] - beam_Times[2*i+1]) < 10:  # When patient failed to hold breath (mistake)
                i += 1
            else:
                modified_beam_Times.append(beam_Times[2*i+1])
                modified_beam_Times.append(beam_Times[2*i+2])
                i += 1
        modified_beam_Times.append(beam_Times[-1])
    return modified_beam_Times

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

"""Reproducibility"""
# Average level per interval
def avg_lvl_per_interval(Amps):
    return np.sum(Amps) / len(Amps)

# Reproducibility (with a list of average levels)
def reproducibility(avg_levels):
    return max(avg_levels) - min(avg_levels)

# Average reproducibility
# def mean_reproducibility(reprods):
#     return np.mean(np.array(reprods))

"""Stability"""
# Vertical Error
def error_per_interval(Amps):
    dt = 0.015
    Times = [t*dt for t in range(len(Amps))]
    slope, _ = np.polyfit(Times, Amps, deg=1)
    duration = dt * ( len(Amps) - 1 )
    return abs(slope) * duration

# Stability (with a list of vertical distances)
def stability(errors):
    return max(errors)

# Average stability
# def mean_stability(stabs):
#     return np.mean(np.array(stabs))

"""Statistics"""
def coeff_var(total_metrics): # CV (coefficients of variation)
    """CV_intra & CV_inter"""
    mean = np.mean(total_metrics)
    std = np.std(total_metrics)
    return std / mean

# 특정 환자의 특정 fraction에 대한 호흡 분석 (전체 field-beam_interval을 평균 낸 결과)
def fraction_analysis(list_of_field_data):
    per_field_levels, per_field_errors = [], []
    for fld_data in list_of_field_data:
        (data_Times, data_Amps), (beam_Times, beam_States) = read_field_data(fld_data)
        beam_Times = beam_modification(beam_Times)
        enabled_intervals, num_intervals = beam_enabling_intervals(data_Times, data_Amps, beam_Times)
        total_intv_levels, total_intv_errors = 0, 0
        for intv in range(num_intervals):
            average_level = avg_lvl_per_interval(enabled_intervals[intv])
            vertical_error = error_per_interval(enabled_intervals[intv])
            total_intv_levels += average_level
            total_intv_errors += vertical_error
        per_field_levels.append( (total_intv_levels / num_intervals) )
        per_field_errors.append( (total_intv_errors / num_intervals) )
    inter_reprod = reproducibility(per_field_levels)
    inter_stab = stability(per_field_errors)
    level_mean, error_mean = np.mean(per_field_levels), np.mean(per_field_errors)
    level_std, error_std = np.std(per_field_levels), np.std(per_field_errors)

    return round(inter_reprod, 4), round(level_mean, 4), round(level_std, 4), round(inter_stab, 4), round(error_mean, 4), round(error_std, 4)

def batch_processing(total_patients):
    total_results = dict()
    for patient_path in total_patients:
        curr_ID = os.path.basename(patient_path)
        patient_results = dict()
        for fx in sorted(os.listdir(patient_path), key=int):  # 20 fractions loop
            fx_path = os.path.join(patient_path, fx)
            per_fraction_data = []
            for fld in sorted(os.listdir(fx_path)):  # Num of fields loop
                fld_data = os.path.join(fx_path, fld)
                per_fraction_data.append(fld_data)
            fx_result = fraction_analysis(per_fraction_data) # Fraction 단위 분석 함수
            patient_results[fx] = fx_result
        total_results[curr_ID] = patient_results

    return total_results

def save_results(analyzed_results, result_root, data_type):
    """Data_type이 결정되고, 해당 data type의 모든 환자 리스트를 입력으로 받아 모두 분석하는 코드"""
    curr_result_directory = os.path.join(result_root, data_type)
    os.makedirs(curr_result_directory, exist_ok=True)
    for patient_ID in sorted(list(analyzed_results.keys())):
        inter_rpds, inter_stbs = [], []
        level_means, error_means = [], []
        level_stds, error_stds = [], []
        level_cvs, error_cvs = [], []
        curr_ID = patient_ID.split("\\")[-1]
        curr_fractions = sorted(list(analyzed_results[patient_ID].keys()), key=int)
        patient_results = pd.DataFrame()
        for fraction in curr_fractions:
            inter_reprod, lvl_mean, lvl_std, inter_stab, err_mean, err_std = analyzed_results[patient_ID][fraction]
            inter_rpds.append(inter_reprod)
            inter_stbs.append(inter_stab)
            level_means.append(lvl_mean)
            level_stds.append(lvl_std)
            level_cvs.append(lvl_std / lvl_mean)
            error_means.append(err_mean)
            error_stds.append(err_std)
            error_cvs.append(err_std / err_mean)
        patient_results["Inter_RPD"] = inter_rpds
        patient_results["LVL_Mean"] = level_means
        patient_results["LVL_STD"] = level_stds
        patient_results["LVL_CV"] = level_cvs
        patient_results["Inter_STB"] = inter_stbs
        patient_results["VD_Mean"] = error_means
        patient_results["VD_STD"] = error_stds
        patient_results["VD_CV"] = error_cvs
        curr_result_filename = f"{curr_result_directory}/{curr_ID}.xlsx"
        patient_results.to_excel(curr_result_filename)
    return

"""Check if there's any correlation between fractions"""
def plot_by_fx(patient_data, plot_root):
    datatype = os.path.dirname(patient_data).split("\\")[-1]
    dir_by_datatype = os.path.join(plot_root, datatype)
    os.makedirs(dir_by_datatype, exist_ok=True)
    patient_ID = os.path.basename(patient_data).split("_")[0]
    df = pd.read_excel(patient_data)
    lvl_mean = df["LVL_Mean"]
    lvl_std = df["LVL_STD"]
    lvl_cv = df["LVL_CV"]
    vd_mean = df["VD_Mean"]
    vd_std = df["VD_STD"]
    vd_cv = df["VD_CV"]
    x = range(len(lvl_std))
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    fig.suptitle("Metrics over fractions")
    axes[0][0].plot(x, lvl_mean), axes[0][0].set_title("Average Level (Mean)")
    axes[0][1].plot(x, lvl_std), axes[0][1].set_title("Average Level (STD)")
    axes[0][2].plot(x, lvl_cv), axes[0][2].set_title("Average Level (CV)")
    axes[1][0].plot(x, vd_mean), axes[1][0].set_title("Vertical Distance (Mean)")
    axes[1][1].plot(x, vd_std), axes[1][1].set_title("Vertical Distance (STD)")
    axes[1][2].plot(x, vd_cv), axes[1][2].set_title("Vertical Distance (CV)")
    filename_to_save = os.path.join(dir_by_datatype, f"{patient_ID}.jpg")
    fig.savefig(filename_to_save)
    plt.close()
    return