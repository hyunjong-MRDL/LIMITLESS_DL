import data, models, roc
import torch, os, random, PIL
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from utils import seed_everything, CFG

seed_everything(CFG["SEED"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = "D:\\Datasets\\SJS\\"
processed_root = os.path.join(data_root, "Processed")
plot_root = "E:\\Results\\SJS\\Figures\\"
os.makedirs(plot_root, exist_ok=True)

excel_root = "C:\\Users\\PC00\\Desktop\\SJS\\"
os.makedirs(excel_root, exist_ok=True)

input("Press Enter to start analysis: ")
print()
while True:
    control_group = input("Select type of control group: ").strip().upper()
    print()
    if control_group == ("NORMAL" or "NONSJS"): break

excel_save_path = os.path.join(excel_root, f"{control_group}_SJS.xlsx")

SJS_path = data.path_by_diagnosis(processed_root, "SJS")
CTR_path = data.path_by_diagnosis(processed_root, control_group)
SJS_ID = data.ID_summary(SJS_path)
CTR_ID = data.ID_summary(CTR_path)

test_SJS_idx = random.sample(range(len(SJS_ID)), len(SJS_ID)//2)
test_CTR_idx = random.sample(range(len(CTR_ID)), len(CTR_ID)//2)
test_SJS_ID = [SJS_ID[i] for i in test_SJS_idx]
test_CTR_ID = [CTR_ID[i] for i in test_CTR_idx]

test_SJS_path = data.path_by_IDs(test_SJS_ID, SJS_path)
test_CTR_path = data.path_by_IDs(test_CTR_ID, CTR_path)

total_ID = test_SJS_ID + test_CTR_ID
total_path = test_SJS_path + test_CTR_path

resize = transforms.Compose([
    transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    transforms.ToTensor()
])

def roc_curve(preds, labels):
    tpr, fpr = [], []
    best_acc = 0

    thresholds = np.linspace(-1, 2, 100)
    for thresh in thresholds:
        tpr.append(roc.true_positive_rate(preds, labels, thresh))
        fpr.append(roc.false_positive_rate(preds, labels, thresh))
        acc = roc.accuracy(preds, labels, thresh)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    score = roc.auc_score(tpr, fpr)

    return best_acc, best_thresh

"""나중에 test 결과 확인 할 때"""
# (1) 현재 이미지의 ID 추출
# (2) CTR/SJS 확인 [curr_ID in 'CTR_ID' or 'SJS_ID']
# (3) LABEL이랑 확인 (정답을 맞혔는가)

PG_ResNet = models.ResNet().to(device)
PG_VGG = models.VGG16().to(device)
PG_Inception = models.Inception().to(device)
SG_ResNet = models.ResNet().to(device)
SG_VGG = models.VGG16().to(device)
SG_Inception = models.Inception().to(device)
PG_ResNet.load_state_dict(torch.load(CFG["pg_res_path"], map_location="cuda"))
PG_VGG.load_state_dict(torch.load(CFG["pg_vgg_path"], map_location="cuda"))
PG_Inception.load_state_dict(torch.load(CFG["pg_inc_path"], map_location="cuda"))
SG_ResNet.load_state_dict(torch.load(CFG["sg_res_path"], map_location="cuda"))
SG_VGG.load_state_dict(torch.load(CFG["sg_vgg_path"], map_location="cuda"))
SG_Inception.load_state_dict(torch.load(CFG["sg_inc_path"], map_location="cuda"))
PG_ResNet.eval()
PG_VGG.eval()
PG_Inception.eval()
SG_ResNet.eval()
SG_VGG.eval()
SG_Inception.eval()

# results = dict()
total_preds, total_labels = [], []
for idx, ID in enumerate(total_ID):
    curr_datalist = total_path[idx]
    curr_label = "CTR" if ID in CTR_ID else "SJS"
    output_list, model_list, gland_list = [], [], []
    for datapath in curr_datalist:
        curr_data = resize(PIL.Image.open(datapath)).to(torch.float32).unsqueeze(0).to(device)
        data_gland = datapath.split("\\")[-1].split("_")[0]
        gland_list.append(data_gland)
        with torch.no_grad():
            if data_gland == "PTG":
                res_out = PG_ResNet(curr_data)[0][0].detach().cpu().item()
                vgg_out = PG_VGG(curr_data)[0][0].detach().cpu().item()
                inc_out = PG_Inception(curr_data)[0][0].detach().cpu().item()
                max_output = np.max([res_out, vgg_out, inc_out])
                max_model = np.argmax([res_out, vgg_out, inc_out])
                output_list.append(max_output)
                model_list.append(max_model)
            else:
                res_out = SG_ResNet(curr_data)[0][0].detach().cpu().item()
                vgg_out = SG_VGG(curr_data)[0][0].detach().cpu().item()
                inc_out = SG_Inception(curr_data)[0][0].detach().cpu().item()
                max_output = np.max([res_out, vgg_out, inc_out])
                max_model = np.argmax([res_out, vgg_out, inc_out])
                output_list.append(max_output)
                model_list.append(max_model)
    curr_max_idx = np.argmax(output_list)
    total_preds.append(output_list[curr_max_idx])
    total_labels.append(curr_label)
    """정답을 맞힌 경우"""
    # if (curr_label == "CTR") and (output_list[curr_max_idx] < threshold):
    #     results[ID] = curr_datalist[curr_max_idx]
    # elif (curr_label == "SJS") and (output_list[curr_max_idx] < threshold):
    #     results[ID] = curr_datalist[curr_max_idx]

best_acc, best_thresh = roc_curve(total_preds, total_labels)
print(f"ACC: {best_acc} at {best_thresh}")