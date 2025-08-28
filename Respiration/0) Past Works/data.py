import os

"""Patient path"""
"""Treatment type: (STATIC or ARC)"""
"""Breath type: (Breathhold or FULL)"""
def patient_path(path, treatment_type, breath_type):
    path = f"{path}{treatment_type}_treatment/"
    for folder in os.listdir(path):
        if breath_type in folder: path = f"{path}{folder}/"
        else: continue
    return path, len(os.listdir(path))

"""Fraction path"""
"""1 fraction: Whole session during ONE day"""
def fraction_path(path, fraction):
    path = f"{path}{fraction}/"
    return path, len(os.listdir(path))

"""Total data (AP, Beams)"""
"""One sequence per field"""
def read_field_data(fraction_path, field):
    filepath = f"{fraction_path}{os.listdir(fraction_path)[field-1]}"
    data_Times, data_Amps = [], []
    beam_Times, beam_States = [], []
    THICK_cnt, thin_cnt = 0, 0
    data_flag, beam_flag  = False, False

    with open(filepath, "r") as file: # Open file
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
                    data_Amps.append(float(amplitude))
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