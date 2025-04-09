import os
import numpy as np
import matplotlib.pyplot as plt
from data_new import read_field_data, beam_modification, beam_enabling_intervals, error_per_interval

plot_root = "E:\\Results\\Respiration\\MTG_4\\Session_Plots\\"

def get_beam_std_per_field(field_path):
    (data_Times, data_Amps), (beam_Times, _) = read_field_data(field_path)
    beam_Times = beam_modification(beam_Times)
    enabled_intervals, num_intervals = beam_enabling_intervals(data_Times, data_Amps, beam_Times)
    stds = []
    for intv in range(num_intervals):
        stds.append(np.std(enabled_intervals[intv]))
    return stds

def get_beam_vd_per_field(field_path):
    (data_Times, data_Amps), (beam_Times, _) = read_field_data(field_path)
    beam_Times = beam_modification(beam_Times)
    enabled_intervals, num_intervals = beam_enabling_intervals(data_Times, data_Amps, beam_Times)
    vds = []
    for intv in range(num_intervals):
        vds.append(error_per_interval(enabled_intervals[intv]))
    return vds

def analyze_patient(patient_path):
    all_fraction_beam_stds = []
    all_fraction_beam_vds = []

    for fraction_name in sorted(os.listdir(patient_path)):
        fx_path = os.path.join(patient_path, fraction_name)
        if not os.path.isdir(fx_path):
            continue

        fraction_beam_stds = []
        fraction_beam_vds = []

        for field_name in sorted(os.listdir(fx_path)):
            field_path = os.path.join(fx_path, field_name)
            if not os.path.isfile(field_path):
                continue
            
            stds = get_beam_std_per_field(field_path)
            vds = get_beam_vd_per_field(field_path)
            if stds:
                fraction_beam_stds.append(stds)
            if vds:
                fraction_beam_vds.append(vds)

        if fraction_beam_stds:
            flat = [item for sublist in fraction_beam_stds for item in sublist]
            all_fraction_beam_stds.append(flat)
        if fraction_beam_vds:
            flat = [item for sublist in fraction_beam_vds for item in sublist]
            all_fraction_beam_vds.append(flat)

    return all_fraction_beam_stds, all_fraction_beam_vds

def plot_beam_STDs(patient_path):
    patient_id = os.path.basename(patient_path)
    datatype = patient_path.split("\\")[-2]
    curr_save_path = os.path.join(plot_root, "STD_Analysis", datatype)
    os.makedirs(curr_save_path, exist_ok=True)
    beam_stds_per_fraction, _ = analyze_patient(patient_path)

    if not beam_stds_per_fraction:
        print(f"{patient_id}: No valid data found.")
        return

    plt.figure(figsize=(10, 6))
    for i, std_list in enumerate(beam_stds_per_fraction):
        plt.plot(range(len(std_list)), std_list, label=f"Fraction {i+1}", marker='o')

    plt.title(f"{patient_id}")
    plt.xlabel("Beam Session Index")
    plt.ylabel("STD (mm)")
    plt.legend(title="Fractions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(curr_save_path, f"{patient_id}.jpg"))
    plt.close()

def plot_beam_VDs(patient_path):
    patient_id = os.path.basename(patient_path)
    datatype = patient_path.split("\\")[-2]
    curr_save_path = os.path.join(plot_root, "VD_Analysis", datatype)
    os.makedirs(curr_save_path, exist_ok=True)
    _, beam_vds_per_fraction = analyze_patient(patient_path)

    if not beam_vds_per_fraction:
        print(f"{patient_id}: No valid data found.")
        return

    plt.figure(figsize=(10, 6))
    for i, vd_list in enumerate(beam_vds_per_fraction):
        plt.plot(range(len(vd_list)), vd_list, label=f"Fraction {i+1}", marker='o')

    plt.title(f"{patient_id}")
    plt.xlabel("Beam Session Index")
    plt.ylabel("VD (mm)")
    plt.legend(title="Fractions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(curr_save_path, f"{patient_id}.jpg"))
    plt.close()

def analyze_all_patients(root_path):
    for patient_name in sorted(os.listdir(root_path)):
        patient_path = os.path.join(root_path, patient_name)
        if os.path.isdir(patient_path):
            print(f"Processing patient: {patient_name}")
            plot_beam_STDs(patient_path)
            plot_beam_VDs(patient_path)

analyze_all_patients("D:\\Datasets\\Respiration\\HCC\\non-education\\")
analyze_all_patients("D:\\Datasets\\Respiration\\HCC\\education\\")
analyze_all_patients("D:\\Datasets\\Respiration\\Mammo\\")