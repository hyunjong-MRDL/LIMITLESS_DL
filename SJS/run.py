import os, random, torch
import numpy as np
from collections import defaultdict
import data, models, utils
from utils import seed_everything, CFG

seed_everything(CFG["SEED"])

device = "cuda" if torch.cuda.is_available() else 'cpu'

root = root = "D:/Datasets/SJS/"
processed_root = f"{root}Processed/"

mode = input("Select run MODE (train/test) : ")

total_SJS = data.split_by_patient(data.load_by_diagnosis(processed_root, "SJS"))
total_CTR = data.split_by_patient(data.load_by_diagnosis(processed_root, CFG["CONTROL"]))

train_SJS, test_SJS = data.train_test_split(total_SJS, CFG["TEST_PORTION"])
train_CTR, test_CTR = data.train_test_split(total_CTR, CFG["TEST_PORTION"])

model = models.FusionModel().to(device)
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["LR"])

if mode == "train":
    train_pg, train_sg, train_y = data.gland_dataset(train_SJS, train_CTR)
    total_correct = utils.train(train_pg, train_sg, train_y, model, criterion, optimizer)
else:
    test_pg, test_sg, test_y = data.gland_dataset(test_SJS, test_CTR)
    preds, gts = utils.test(test_pg, test_sg, test_y, model)
    utils.ROC(preds, gts, len(total_SJS), len(total_CTR))