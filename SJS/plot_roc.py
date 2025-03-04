import os
from data import read_xls_data
from roc import roc_curve

root = "D:/Datasets/SJS/Processed/"

NORMAL_SJS, NONSJS_SJS = read_xls_data("./Resident_preds.xlsx")

NORMAL_ID = list(NORMAL_SJS["ID"])
NONSJS_ID = list(NONSJS_SJS["ID"])

"""NORMAL_SJS"""
resident1 = list(NORMAL_SJS["diag1"])
resident2 = list(NORMAL_SJS["diag2"])
resident3 = list(NORMAL_SJS["diag3"])

"""NONSJS_SJS"""
resident4 = list(NONSJS_SJS["diag4"])
resident5 = list(NONSJS_SJS["diag5"])
resident6 = list(NONSJS_SJS["diag6"])

sjs_id, nonsjs_id, normal_id = [], [], []
for sjs in os.listdir(f"{root}SJS/"):
    ID = sjs.split("_")[1].split("-")[0]
    if ID not in sjs_id: sjs_id.append(int(ID))
for nonsjs in os.listdir(f"{root}NON_SJS/"):
    ID = nonsjs.split("_")[1].split("-")[0]
    if ID not in nonsjs_id: nonsjs_id.append(int(ID))
for normal in os.listdir(f"{root}NORMAL/"):
    ID = normal.split("_")[1].split("-")[0]
    if ID not in normal_id: normal_id.append(int(ID))

NORMAL_label = []
for ID in NORMAL_ID:
    if ID in sjs_id: NORMAL_label.append(1)
    elif ID in normal_id: NORMAL_label.append(0)

NONSJS_label = []
for ID in NONSJS_ID:
    if ID in sjs_id: NONSJS_label.append(1)
    elif ID in nonsjs_id: NONSJS_label.append(0)

"""NORMAL_SJS"""
preds1 = []
for pred in resident1:
    if pred == 1: preds1.append(0)
    elif pred == 2: preds1.append(0.33)
    elif pred == 3: preds1.append(0.66)
    else: preds1.append(1)

preds2 = []
for pred in resident2:
    if pred == 1: preds2.append(0)
    elif pred == 2: preds2.append(0.33)
    elif pred == 3: preds2.append(0.66)
    else: preds2.append(1)

preds3 = []
for pred in resident3:
    if pred == 1: preds3.append(0)
    elif pred == 2: preds3.append(0.33)
    elif pred == 3: preds3.append(0.66)
    else: preds3.append(1)

"""NONSJS_SJS"""
preds4 = []
for pred in resident4:
    if pred == 1: preds4.append(0)
    elif pred == 2: preds4.append(0.33)
    elif pred == 3: preds4.append(0.66)
    else: preds4.append(1)

preds5 = []
for pred in resident5:
    if pred == 1: preds5.append(0)
    elif pred == 2: preds5.append(0.33)
    elif pred == 3: preds5.append(0.66)
    else: preds5.append(1)

preds6 = []
for pred in resident6:
    if pred == 1: preds6.append(0)
    elif pred == 2: preds6.append(0.33)
    elif pred == 3: preds6.append(0.66)
    else: preds6.append(1)

if __name__ == "__main__":
    roc_curve(preds1, NORMAL_label, 1)
    roc_curve(preds2, NORMAL_label, 2)
    roc_curve(preds3, NORMAL_label, 3)

    roc_curve(preds4, NONSJS_label, 4)
    roc_curve(preds5, NONSJS_label, 5)
    roc_curve(preds6, NONSJS_label, 6)