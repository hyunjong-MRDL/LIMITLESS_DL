import torch
import data, models, utils
from utils import seed_everything, CFG, control_group

seed_everything(CFG["SEED"])

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)

root = root = "D:/Datasets/SJS/"
processed_root = f"{root}Processed/"

total_SJS = data.split_by_patient(data.load_by_diagnosis(processed_root, "SJS"))
total_CTR = data.split_by_patient(data.load_by_diagnosis(processed_root, control_group))

train_SJS, test_SJS = data.train_test_split(total_SJS, CFG["TEST_PORTION"])
train_CTR, test_CTR = data.train_test_split(total_CTR, CFG["TEST_PORTION"])

PG_ResNet = models.ResNet().to(device)
PG_VGG = models.VGG16().to(device)
PG_Inception = models.Inception().to(device)

SG_ResNet = models.ResNet().to(device)
SG_VGG = models.VGG16().to(device)
SG_Inception = models.Inception().to(device)

X_pg, X_sg, Y_pg, Y_sg = data.gland_dataset(test_SJS, test_CTR)
preds, gts = utils.test_2(X_pg, X_sg, Y_pg, Y_sg, PG_ResNet, PG_VGG, PG_Inception, SG_ResNet, SG_VGG, SG_Inception)
pat_preds, pat_gts = utils.pred_gt_by_patients(preds, gts)
preds, gts = utils.preds_and_gts(pat_preds, pat_gts)

utils.ROC(preds, gts, len(test_SJS), len(test_CTR))