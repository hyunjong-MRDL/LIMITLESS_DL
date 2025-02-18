import torch, PIL
import data, models, utils
from utils import seed_everything, CFG, control_group

seed_everything(CFG["SEED"])

device = "cuda" if torch.cuda.is_available() else 'cpu'

root = "D:/Datasets/SJS/"
processed_root = f"{root}Processed/"
plot_root = "E:/LIMITLESS_DL/SJS/Plot/"
max_img_path = "C:/Users/PC00/Desktop/SJS max_output images.txt"
max_model_path = "C:/Users/PC00/Desktop/SJS max_models.txt"

total_SJS = data.split_by_patient(data.load_by_diagnosis(processed_root, "SJS"))
total_CTR = data.split_by_patient(data.load_by_diagnosis(processed_root, control_group))

train_SJS, test_SJS = data.train_test_split(total_SJS, CFG["TEST_PORTION"])
train_CTR, test_CTR = data.train_test_split(total_CTR, CFG["TEST_PORTION"])

selected_images = data.read_max_img_paths(max_img_path)
selected_models = data.read_max_models(max_model_path)

PG_ResNet = models.ResNet().to(device)
PG_VGG = models.VGG16().to(device)
PG_Inception = models.Inception().to(device)

SG_ResNet = models.ResNet().to(device)
SG_VGG = models.VGG16().to(device)
SG_Inception = models.Inception().to(device)

X_pg, X_sg, Y_pg, Y_sg = data.gland_dataset(test_SJS, test_CTR)
preds, gts, max_models = utils.test(X_pg, X_sg, Y_pg, Y_sg, PG_ResNet, PG_VGG, PG_Inception, SG_ResNet, SG_VGG, SG_Inception)
# pat_preds, pat_gts, pat_images = utils.pred_gt_by_patients(preds, gts)
# preds, gts = utils.preds_and_gts(pat_preds, pat_gts)
# utils.ROC(preds, gts, len(test_SJS), len(test_CTR))

utils.createFolder(plot_root)

for image in selected_images:
    diagnosis = image.split("/")[4]
    gland = image.split("/")[-1].split("_")[0]
    ID = image.split("/")[-1].split("_")[1].split("-")[0]
    selected_model = selected_models[ID]

    if gland == "PTG":
        if selected_model == "ResNet":
            model = PG_ResNet
        elif selected_model == "VGG":
            model = PG_VGG
        else: continue
    elif gland == "SMG":
        if selected_model == "ResNet":
            model = SG_ResNet
        elif selected_model == "VGG":
            model = SG_VGG
        else: continue