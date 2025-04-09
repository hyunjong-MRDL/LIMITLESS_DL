import os, data_new
import numpy as np
import matplotlib.pyplot as plt

data_root = "D:\\Datasets\\Respiration\\"
result_root = "E:\\Results\\Respiration\\"

print()
print("호흡 데이터 분석을 시작합니다.")
print()

meeting_number = int(input("미팅 회차를 명시해주세요: "))
print()

plot_root = os.path.join(result_root, f"MTG_{meeting_number}", "Session_Plots")
STD_save_path = os.path.join(plot_root, "STD_Analysis")
VD_save_path = os.path.join(plot_root, "VD_Analysis")
os.makedirs(STD_save_path, exist_ok=True)
os.makedirs(VD_save_path, exist_ok=True)
del result_root, meeting_number

total_list = data_new.patient_listing(data_root)
datatype_list = list(total_list.keys())
del data_root

print("=====현재 총 데이터=====")
for i, data_type in enumerate(datatype_list):
    print(f"{i+1}: {data_type}\t")
del i, data_type
print()

while True:
    type_index = int(input("분석하고 싶은 데이터를 선택해주세요: "))
    print()
    selected_type = datatype_list[type_index - 1]
    proceed = input(f"{selected_type}를 선택하셨습니다. 분석을 진행하고 싶으면 'O', 다른 데이터를 분석하고 싶으면 'X'를 입력해주십시오: ").upper()
    if proceed == "O": break
del datatype_list, type_index, proceed
print()

STD_path_of_type = os.path.join(STD_save_path, selected_type)
VD_path_of_type = os.path.join(VD_save_path, selected_type)
patient_list = total_list[selected_type]
del total_list, selected_type, STD_save_path, VD_save_path

"""현재 Data directory 상태 확인"""
print(f"선택하신 데이터에는 총 {len(patient_list)}명의 환자가 있습니다.")
print()
for patient_dir in patient_list:
    patient_ID = os.path.basename(patient_dir)
    for fx in os.listdir(patient_dir):
        fx_dir = os.path.join(patient_dir, fx)
        VDs_in_fx = []
        for fld in os.listdir(fx_dir):
            fld_data = os.path.join(fx_dir, fld)
            (data_Times, data_Amps), (beam_Times, _) = data_new.read_field_data(fld_data)
            beam_Times = data_new.beam_modification(beam_Times)
            enabled_intervals, num_intervals = data_new.beam_enabling_intervals(data_Times, data_Amps, beam_Times)
            for intv in range(num_intervals):
                VD = data_new.error_per_interval(enabled_intervals[intv])
                VDs_in_fx.append(VD)
        t = range(len(VDs_in_fx))
        curr_patient_save_path = os.path.join(curr_type_save_path, patient_ID)
        os.makedirs(curr_patient_save_path, exist_ok=True)
        plt.figure(figsize=(12, 8)), plt.plot(t, VDs_in_fx), plt.title(f"Fraction{fx}"), plt.savefig(os.path.join(curr_patient_save_path, f"fx{fx}.jpg")), plt.close()