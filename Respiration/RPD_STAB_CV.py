import os, data_new
import numpy as np
import pandas as pd

result_root = "E:\\Results\\Respiration\\"

meeting_number = 4
curr_result_path = os.path.join(result_root, f"MTG_{meeting_number}")
del result_root, meeting_number

organized_data_path = os.path.join(curr_result_path, "Organized_Data")
analysis_save_path = os.path.join(curr_result_path, "Analysis")
os.makedirs(analysis_save_path, exist_ok=True)
del curr_result_path

print("호흡 데이터 분석을 시작합니다.\n")
for datatype in os.listdir(organized_data_path):
    print(f"=== 현재 {datatype} 데이터를 분석 중입니다 ===")
    print()
    total_results = dict()
    total_patients, total_reprods, total_stabs = [], [], []
    total_mean_LVL, total_std_LVL, total_CV_LVL = [], [], []
    total_mean_VD, total_std_VD, total_CV_VD = [], [], []

    curr_type_data_path = os.path.join(organized_data_path, datatype)
    for patient_name in os.listdir(curr_type_data_path):
        total_patients.append(patient_name.split(".")[0])

        curr_data_filename = os.path.join(curr_type_data_path, patient_name)
        curr_data = pd.read_excel(curr_data_filename)

        # 각 fraction별 모든 field에 대한 LVL, VD의 평균 및 표준편차
        LVL_means, VD_means = list(curr_data["LVL Mean"]), list(curr_data["VD Mean"])

        # 모든 fraction에 대한 LVL, VD에 기반한 Reproducibility 및 Stability 계산
        reprod = data_new.reproducibility(LVL_means)
        total_reprods.append( round(reprod, 4) )
        stab = data_new.stability(VD_means)
        total_stabs.append( round(stab, 4) )

        # 모든 fraction에 걸친 LVL 및 VD의 평균 및 표준편차
        mean_LVL, mean_VD = np.mean(LVL_means).item(), np.mean(VD_means).item()
        std_LVL, std_VD = np.std(LVL_means).item(), np.std(VD_means).item()
        total_mean_LVL.append( round(mean_LVL, 4) )
        total_std_LVL.append( round(std_LVL, 4) )
        total_CV_LVL.append( round(std_LVL/mean_LVL, 4) )
        total_mean_VD.append( round(mean_VD, 4) )
        total_std_VD.append( round(std_VD, 4) )
        total_CV_VD.append( round(std_VD/mean_VD, 4) )
    total_results["Patient"] = total_patients
    total_results["Reproducibility"] = total_reprods
    total_results["Stability"] = total_stabs
    total_results["Mean_LVL"] = total_mean_LVL
    total_results["STD_LVL"] = total_std_LVL
    total_results["CV_LVL"] = total_CV_LVL
    total_results["Mean_VD"] = total_mean_VD
    total_results["STD_VD"] = total_std_VD
    total_results["CV_VD"] = total_CV_VD
    
    total_results_filename = os.path.join(analysis_save_path, f"{datatype}_Analysis.xlsx")
    pd.DataFrame(total_results).to_excel(total_results_filename)