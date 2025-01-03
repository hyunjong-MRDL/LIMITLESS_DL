import numpy as np
import matplotlib.pyplot as plt
from utils import createFolder, extract_data, extract_beam_moments

def save_extracted_AP(filepath):
    file_to_read = open(filepath, "r")
    file_type = "Breathhold" if "Breath" in filepath else "Full"

    ID = filepath.split("/")[3]
    field = filepath.split("_")[-1].split(".")[0]
    createFolder(f"./Processed/{file_type}/Patient{ID}")
    file_to_write = open(f"./Processed/{file_type}/Patient{ID}/{field}.txt", "w")
    anterior_count, start_flag = 0, False
    while True:
        line = file_to_read.readline()
        if start_flag and (line == "\n"): break # Breaking condition
        if start_flag:
            time, amplitude = line.strip().split("\t")
            file_to_write.write(f"{time}\t{amplitude}\n")
        elif "Anterior-Posterior" in line: # Count Ant-Post
            anterior_count += 1
        if file_type == "Breathhold":
            if (anterior_count > 1) and ("Amplitude" in line): start_flag = True # Numerical data starts here
        elif file_type == "Full":
            if (anterior_count > 0) and ("Amplitude" in line): start_flag = True
    file_to_read.close()
    file_to_write.close()

    return

def merge_data(dict):
    id_list = list(dict.keys())
    data_type = "Breathhold" if (len(dict[id_list[0]]) > 2) else "Full"
    for ID in id_list:
        data_path = dict[ID]
        time_list, amp_list = [], []
        beamtime_list, state_list = [], []
        createFolder(f"./Merged/{data_type}/{ID}")
        datafile_to_write = open(f"./Merged/{data_type}/{ID}/merged{ID}.txt", "w")
        beamfile_to_write = open(f"./Merged/{data_type}/{ID}/beam{ID}.txt", "w")
        period_of_prev_field = 0.0 if ( len(time_list) == 0 ) else (time_list[-1]+0.015)
        for data in data_path:
            time, amplitude = extract_data(data)
            beamtime, state = extract_beam_moments(data)
            time_list.append(float(time) + period_of_prev_field)
            amp_list.append(float(amplitude))
            beamtime_list.append(float(beamtime) + period_of_prev_field)
            state_list.append(state)
        datafile_to_write.close()
        beamfile_to_write.close()

    return time_list, amp_list

def plot_by_dict(dict):
    id_list = list(dict.keys())
    dict_type = "Breathhold" if (len(dict[id_list[0]]) > 2) else "Full"
    for ID in id_list:
        data_path = dict[ID]
        createFolder(f"./Plots/{dict_type}/Patient{ID}")
        for i in range(len(data_path)):
            if dict_type == "Breathhold":
                fig = plt.figure(figsize=(14,14))
            elif dict_type == "Full":
                fig = plt.figure(figsize=(20,14))
            ax = fig.add_subplot(111)
            time, amplitude = extract_data(data_path[i])
            ax.plot(time, amplitude),
            ax.axis([np.min(np.array(time)), np.max(np.array(time)), np.min(np.array(amplitude)), np.max(np.array(amplitude))]), # plt.axis([xmin, xmax, ymin, ymax])
            ax.set_xlabel("Time (s)"), ax.set_ylabel("Amplitude (cm)")
            ax.set_title(data_path[i].split('_')[-1].split('.')[0])
            fig.savefig(f"./Plots/{dict_type}/Patient{ID}/{data_path[i].split('_')[-1].split('.')[0]}.jpg")
            plt.close()
    return