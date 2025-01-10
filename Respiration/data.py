import os

def patient_path(path, treatment_type, breath_type):
    path = f"{path}{treatment_type}_treatment/"
    for folder in os.listdir(path):
        if breath_type in folder: path = f"{path}{folder}/"
        else: continue
    return path, len(os.listdir(path))

def fraction_path(path, fraction):
    path = f"{path}{fraction}/"
    return path, len(os.listdir(path))

def read_field_AP(fraction_path, field):
    filepath = f"{fraction_path}{os.listdir(fraction_path)[field-1]}"
    data_Times, data_Amps = [], []
    data_type = "Free-breath" if "FULL" in filepath else "Breathhold"
    anterior_count, start_flag = 0, False

    with open(filepath, "r") as file: # Open file
        for line in file:
            if start_flag and line == "\n": break
            if start_flag:
                time, amplitude = line.strip().split("\t")
                data_Times.append(float(time))
                data_Amps.append(float(amplitude))
            elif "Anterior-Posterior" in line:
                anterior_count += 1
            if (data_type == "Breathhold"): # SAME for "ARC type"
                if (anterior_count > 1) and ("Amplitude" in line): start_flag = True
            elif data_type == "Free-breath":
                if (anterior_count > 0) and ("Amplitude" in line): start_flag = True

    return data_Times, data_Amps

'''Extract beam moments (the timing when the beams are enabled)'''
'''Returns (time, beam state)'''
def read_field_beams(fraction_path, field):
    filepath = f"{fraction_path}{os.listdir(fraction_path)[field-1]}"
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