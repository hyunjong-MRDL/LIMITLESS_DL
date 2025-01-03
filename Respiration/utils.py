import os

'''TOTAL DICTIONARY'''
# dict = {"Scan number (trial)": [field1, field2, ..., field9]}
# Scan number: 몇 번째 촬영인지 (총 10회 촬영)
# Field_N: Beam enable moment 기준으로 나눈 series (촬영 1회당 9개의 series)
def construct_total_dict(root):
    breathhold_dict, full_dict = {}, {}
    for data_type in os.listdir(root):
        subpath = root + data_type + "/"
        for scan_num in os.listdir(subpath):
            scan_path = subpath + scan_num + "/"
            scan_data = []
            for data in os.listdir(scan_path):
                data_path = scan_path + data
                scan_data.append(data_path)
            if "FULL" in data_type:
                full_dict[scan_num] = scan_data
            elif "Breathhold" in data_type:
                breathhold_dict[scan_num] = scan_data

    return breathhold_dict, full_dict

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    return

'''Receives exact FILEPATH as argument (extract data of 1 field at the same time)'''
'''Returns (time, amplitude)'''
def extract_data(filepath):
    time_list, amp_list = [], []
    data_type = "Breathhold" if "Breath" in filepath else "Full"
    anterior_count, start_flag = 0, False
    with open(filepath, "r") as file:
        for line in file:
            if start_flag and (line == "\n"): break
            if start_flag:
                time, amplitude = line.strip().split("\t")
                time_list.append(float(time))
                amp_list.append(float(amplitude))
            elif "Anterior-Posterior" in line:
                anterior_count += 1
            if data_type == "Breathhold":
                if (anterior_count > 1) and ("Amplitude" in line): start_flag = True
            elif data_type == "Full":
                if (anterior_count > 0) and ("Amplitude" in line): start_flag = True

    return time_list, amp_list

'''Extract beam moments (the timing when the beams are enabled)'''
'''Returns (time, beam state)'''
def extract_beam_moments(filepath):
    time_list, state_list = [], []
    beam_flag, start_flag = False, False
    with open(filepath, "r") as file:
        for line in file:
            if (beam_flag and start_flag) and line == "\n": break
            elif (beam_flag and start_flag):
                time, state = line.strip().split("\t")
                time_list.append(float(time))
                state_list.append(int(state))
            elif "Beam Enable/Disable Moments" in line: beam_flag = True
            elif beam_flag and ("Enable" in line): start_flag = True

    return time_list, state_list