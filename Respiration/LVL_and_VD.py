import os, data_new

"""Data Root 디렉토리에 있는 모든 데이터를 한꺼번에 분석해서 Excel파일로 저장하는 코드"""

data_root = "D:\\Datasets\\Respiration\\"
result_root = "E:\\Results\\Respiration\\"

print()
print("호흡 데이터 분석을 시작합니다.")
print()

meeting_number = int(input("미팅 회차를 명시해주세요: "))
print()

curr_result_path = os.path.join(result_root, f"MTG_{meeting_number}")
os.makedirs(curr_result_path, exist_ok=True)
del result_root, meeting_number

organized_save_path = os.path.join(curr_result_path, "Organized_Data")
os.makedirs(organized_save_path, exist_ok=True)
del curr_result_path

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
    print(f"{selected_type}를 선택하셨습니다.")
    print()
    proceed = input("선택한 데이터를 분석하고 싶으면 'O', 다른 데이터를 확인/분석하고 싶으시면 'X'를 입력해주십시오: ").upper()
    if proceed == "O": break
del datatype_list, type_index, proceed
print()

patient_list = total_list[selected_type]
del total_list

"""현재 Data directory 상태 확인"""
print(f"선택하신 데이터에는 총 {len(patient_list)}명의 환자가 있습니다.")
print()

# print(There are (?) fractions for this patient)

total_results = data_new.batch_processing(patient_list)
del patient_list

data_new.save_results(total_results, organized_save_path, selected_type)