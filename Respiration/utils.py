import os, shutil

"""Fraction 데이터 끼리 폴더 생성하고 정리하는 코드"""
## 아직 불안정 -> 정말 필요할 때만 사용하기
def directory_sort_by_fraction(fractions_list):
    for patient_path in fractions_list:
        for directory in os.listdir(patient_path):
            curr_path = os.path.join(patient_path, directory)
            if (".txt" in curr_path) and ("README" not in curr_path):
                curr_fx = curr_path.split("fx")[0]
                fx_path = os.path.join(patient_path, curr_fx)
                os.makedirs(fx_path, exist_ok=True)
                shutil.move(curr_path, fx_path)
    return