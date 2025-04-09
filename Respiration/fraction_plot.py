import os, data_new

result_root = "E:\\Results\\Respiration\\"

meeting_number = 4
curr_result_path = os.path.join(result_root, f"MTG_{meeting_number}")
del result_root, meeting_number

organized_data_path = os.path.join(curr_result_path, "Organized_Data")
plots_save_path = os.path.join(curr_result_path, "Fraction_Plots")
os.makedirs(plots_save_path, exist_ok=True)
del curr_result_path

print("호흡 데이터 분석을 시작합니다.\n")
for datatype in os.listdir(organized_data_path):
    print(f"=== 현재 {datatype} 데이터를 분석 중입니다 ===")
    print()
    curr_type_data_path = os.path.join(organized_data_path, datatype)
    for patient_name in os.listdir(curr_type_data_path):
        curr_data_filename = os.path.join(curr_type_data_path, patient_name)
        data_new.plot_by_fx(curr_data_filename, plots_save_path)

print("모든 데이터 분석이 끝났습니다.\n프로그램을 종료합니다.\n")