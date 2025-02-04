import torch
import data, models, utils
from utils import seed_everything, CFG

seed_everything(CFG["SEED"])

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)

root = root = "D:/Datasets/SJS/"
processed_root = f"{root}Processed/"

total_SJS = data.split_by_patient(data.load_by_diagnosis(processed_root, "SJS"))
total_CTR = data.split_by_patient(data.load_by_diagnosis(processed_root, CFG["CONTROL"]))

train_SJS, test_SJS = data.train_test_split(total_SJS, CFG["TEST_PORTION"])
train_CTR, test_CTR = data.train_test_split(total_CTR, CFG["TEST_PORTION"])

model = models.FusionModel().to(device)

test_pg, test_sg, test_y = data.gland_dataset(test_SJS, test_CTR)
preds, gts = utils.test(test_pg, test_sg, test_y, model)
utils.ROC(preds, gts, len(total_SJS), len(total_CTR))