import os, random
import numpy as np
from collections import defaultdict
from utils import seed_everything, CFG

"""Abbreviations"""
# pt.: patient

def load_by_diagnosis(root, diagnosis):
    diagnosis_path = f"{root}{diagnosis}/"
    diagnosis_data = [] # PTG/SMG combined -> needs to be classified
    for data_path in os.listdir(diagnosis_path):
        diagnosis_data.append(f"{diagnosis_path}{data_path}")
    return diagnosis_data

def split_by_patient(diagnosis_data):
    patient_paths = defaultdict(list)
    for fullname in diagnosis_data:
        data_name = fullname.split("/")[-1] # "~~~.jpg"
        ID = data_name.split("_")[1].split("-")[0]
        patient_paths[ID].append(fullname)
    return patient_paths

def split_by_gland(diagnosis_data):
    patient_paths = defaultdict(list)
    for fullname in diagnosis_data:
        data_name = fullname.split("/")[-1] # "~~~.jpg"
        gland = data_name.split("_")[0]
        patient_paths[gland].append(fullname)
    return patient_paths

def data_summary(patient_paths):
    patient_IDs = list(patient_paths.keys())
    print(f"There are {len(patient_IDs)} patients in total.")
    print("================================================")
    print()
    for ID in patient_IDs:
        print(f"\nID: {ID}")
        patient_data = patient_paths[ID]
        PTG_count, SMG_count = 0, 0
        for fullname in patient_data:
            gland = fullname.split("/")[-1].split("_")[0]
            if gland == "PTG": PTG_count += 1
            else: SMG_count += 1
        print(f"PTG: {PTG_count} files")
        print(f"SMG: {SMG_count} files")
    return

def train_test_split(pt_paths, ratio):
    pt_IDs = list(pt_paths.keys())
    random.shuffle(pt_IDs)
    train_length = int(len(pt_IDs) * ratio)
    train_pt_paths, test_pt_paths = defaultdict(list), defaultdict(list)

    for idx, ID in enumerate(pt_IDs):
        if idx <= train_length:
            for path in pt_paths[ID]:
                train_pt_paths[ID].append(path)
        else:
            for path in pt_paths[ID]:
                test_pt_paths[ID].append(path)

    return train_pt_paths, test_pt_paths

def gland_dataset(SJS_by_patient, CTR_by_patient):
    X_pg, X_sg, Y_pg, Y_sg = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for sjs_ID in SJS_by_patient:
        ID_data = SJS_by_patient[sjs_ID]
        for path in ID_data:
            if "PTG" in path:
                X_pg[sjs_ID].append(path)
                Y_pg[sjs_ID].append(np.array([1]))
            else:
                X_sg[sjs_ID].append(path)
                Y_sg[sjs_ID].append(np.array([1]))
    
    for ctr_ID in CTR_by_patient:
        ID_data = CTR_by_patient[ctr_ID]
        for path in ID_data:
            if "PTG" in path:
                X_pg[ctr_ID].append(path)
                Y_pg[ctr_ID].append(np.array([0]))
            else:
                X_sg[ctr_ID].append(path)
                Y_sg[ctr_ID].append(np.array([0]))

    return X_pg, X_sg, Y_pg, Y_sg