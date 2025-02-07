import PIL, torch, random, os, time
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True

device = "cuda" if torch.cuda.is_available() else 'cpu'

# CFG={'SEED' : 42,  # 42~46
#      'IMG_SIZE' : 256,
#      'TEST_PORTION' : 0.5,  # Test set 비율
#      'CONTROL' : "NORMAL",  # "NORMAL" or "NON_SJS"
#      'save_path' : "E:/model_save_path/SJS/model_fusion/attn_module.pt",
#      'EPOCHS' : 15,
#      'BATCH_SIZE' : 4,
#      'LR' : 1e-4}

control_group = "NON_SJS"  # NORMAL or NON_SJS

CFG={'SEED' : 46,  # 42~46
     'IMG_SIZE' : 224,
     'TEST_PORTION' : 0.5,  # Test set 비율
     'pg_res_path' : f"E:/model_save_path/SJS/{control_group}_save_path/{control_group}_5x_test50(PG_Res)_seed46.pt",
     'pg_vgg_path' : f"E:/model_save_path/SJS/{control_group}_save_path/{control_group}_5x_test50(PG_VGG)_seed46.pt",
     'pg_inc_path' : f"E:/model_save_path/SJS/{control_group}_save_path/{control_group}_5x_test50(PG_Inception)_seed46.pt",
     'sg_res_path' : f"E:/model_save_path/SJS/{control_group}_save_path/{control_group}_5x_test50(SG_Res)_seed46.pt",
     'sg_vgg_path' : f"E:/model_save_path/SJS/{control_group}_save_path/{control_group}_5x_test50(SG_VGG)_seed46.pt",
     'sg_inc_path' : f"E:/model_save_path/SJS/{control_group}_save_path/{control_group}_5x_test50(SG_Inception)_seed46.pt",
     'EPOCHS' : 15,
     'BATCH_SIZE' : 4,
     'LR' : 1e-4}

preprocessing = transforms.Compose([
    transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    transforms.ToTensor()
])

def train(X_pg, X_sg, y, model, criterion, optimizer):
    model.train()
    total_patients = list(y.keys())
    total_preds = defaultdict(list)
    total_loss = []
    for epoch in range(CFG["EPOCHS"]):
        epoch_start = time.time()
        for patient in total_patients:
            pg_data, sg_data = [], []
            labels = y[patient]
            if patient in X_pg.keys():
                pg_data = X_pg[patient]
            if patient in X_sg.keys():
                sg_data = X_sg[patient]
            
            if len(pg_data) > len(sg_data):
                for i in range(len(pg_data)):
                    if i < len(sg_data):
                        curr_pg = preprocessing(PIL.Image.open(pg_data[i]))
                        curr_sg = preprocessing(PIL.Image.open(sg_data[i]))
                    else:
                        curr_pg = preprocessing(PIL.Image.open(pg_data[i]))
                        curr_sg = torch.zeros((3, CFG["IMG_SIZE"], CFG["IMG_SIZE"]))
                    
                    curr_pg = curr_pg.float().unsqueeze(0).to(device)
                    curr_sg = curr_sg.float().unsqueeze(0).to(device)
                    label = torch.from_numpy(labels[i].astype(np.float32)).unsqueeze(0).to(device)

                    output = model(curr_pg, curr_sg)
                    total_preds[patient].append(output.detach().cpu().tolist())
                    loss = criterion(output, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss.append(loss)

            elif len(sg_data) > len(pg_data):
                for i in range(len(sg_data)):
                    if i < len(pg_data):
                        curr_sg = preprocessing(PIL.Image.open(sg_data[i]))
                        curr_pg = preprocessing(PIL.Image.open(pg_data[i]))
                    else:
                        curr_sg = preprocessing(PIL.Image.open(sg_data[i]))
                        curr_pg = torch.zeros((3, CFG["IMG_SIZE"], CFG["IMG_SIZE"]))

                    curr_pg = curr_pg.float().unsqueeze(0).to(device)
                    curr_sg = curr_sg.float().unsqueeze(0).to(device)
                    label = torch.from_numpy(labels[i].astype(np.float32)).unsqueeze(0).to(device)

                    output = model(curr_pg, curr_sg)
                    total_preds[patient].append(output.detach().cpu().tolist())
                    loss = criterion(output, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss.append(loss)
        
        epoch_end = time.time()
        print(f"Epoch{epoch} finished. ({(epoch_end-epoch_start)//60}min {(epoch_end-epoch_start)%60}sec passed)")

    torch.save(model.state_dict(), CFG["save_path"])
    return total_preds

def test(X_pg, X_sg, y, model):
    model.load_state_dict(torch.load(CFG["save_path"], map_location="cuda"))
    model.eval()
    total_patients = list(y.keys())
    preds, gts = defaultdict(list), defaultdict(list)
    for patient in total_patients:
        pg_data, sg_data = [], []
        labels = y[patient]
        if patient in X_pg.keys():
            pg_data = X_pg[patient]
        if patient in X_sg.keys():
            sg_data = X_sg[patient]
        
        if len(pg_data) > len(sg_data):
            for i in range(len(pg_data)):
                if i < len(sg_data):
                    curr_pg = preprocessing(PIL.Image.open(pg_data[i]))
                    curr_sg = preprocessing(PIL.Image.open(sg_data[i]))
                else:
                    curr_pg = preprocessing(PIL.Image.open(pg_data[i]))
                    curr_sg = torch.zeros((3, CFG["IMG_SIZE"], CFG["IMG_SIZE"]))
                
                curr_pg = curr_pg.float().unsqueeze(0).to(device)
                curr_sg = curr_sg.float().unsqueeze(0).to(device)
                label = torch.from_numpy(labels[i].astype(np.float32)).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(curr_pg, curr_sg)
                    preds[patient].append(output.detach().cpu().tolist())
                    gts[patient].append(label.detach().cpu().tolist())

        elif len(sg_data) > len(pg_data):
            for i in range(len(sg_data)):
                if i < len(pg_data):
                    curr_sg = preprocessing(PIL.Image.open(sg_data[i]))
                    curr_pg = preprocessing(PIL.Image.open(pg_data[i]))
                else:
                    curr_sg = preprocessing(PIL.Image.open(sg_data[i]))
                    curr_pg = torch.zeros((3, CFG["IMG_SIZE"], CFG["IMG_SIZE"]))

                curr_pg = curr_pg.float().unsqueeze(0).to(device)
                curr_sg = curr_sg.float().unsqueeze(0).to(device)
                label = torch.from_numpy(labels[i].astype(np.float32)).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(curr_pg, curr_sg)
                    preds[patient].append(output.detach().cpu().tolist())
                    gts[patient].append(label.detach().cpu().tolist())

    return preds, gts

def test_2(X_pg, X_sg, Y_pg, Y_sg, PG_Res, PG_VGG, PG_Inc, SG_Res, SG_VGG, SG_Inc):
    PG_Res.load_state_dict(torch.load(CFG["pg_res_path"], map_location="cuda"))
    PG_VGG.load_state_dict(torch.load(CFG["pg_vgg_path"], map_location="cuda"))
    PG_Inc.load_state_dict(torch.load(CFG["pg_inc_path"], map_location="cuda"))
    SG_Res.load_state_dict(torch.load(CFG["sg_res_path"], map_location="cuda"))
    SG_VGG.load_state_dict(torch.load(CFG["sg_vgg_path"], map_location="cuda"))
    SG_Inc.load_state_dict(torch.load(CFG["sg_inc_path"], map_location="cuda"))
    PG_Res.eval()
    PG_VGG.eval()
    PG_Inc.eval()
    SG_Res.eval()
    SG_VGG.eval()
    SG_Inc.eval()
    pg_patients = list(Y_pg.keys())
    sg_patients = list(Y_sg.keys())
    preds, gts = defaultdict(list), defaultdict(list)
    for patient in pg_patients:
        datas = X_pg[patient]
        labels = Y_pg[patient]
        
        for i in range(len(datas)):
            curr_data = preprocessing(PIL.Image.open(datas[i])).float().unsqueeze(0).to(device)
            curr_label = torch.from_numpy(labels[i].astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                res_out = PG_Res(curr_data)
                vgg_out = PG_VGG(curr_data)
                inc_out = PG_Inc(curr_data)
                output = (res_out + vgg_out + inc_out) / 3
                preds[patient].append(output.detach().cpu().tolist())
                gts[patient].append(curr_label.detach().cpu().tolist())
    
    for patient in sg_patients:
        datas = X_sg[patient]
        labels = Y_sg[patient]

        for i in range(len(datas)):
            curr_data = preprocessing(PIL.Image.open(datas[i])).float().unsqueeze(0).to(device)
            label = torch.from_numpy(labels[i].astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                res_out = SG_Res(curr_data)
                vgg_out = SG_VGG(curr_data)
                inc_out = SG_Inc(curr_data)
                output = (res_out + vgg_out + inc_out) / 3
                preds[patient].append(output.detach().cpu().tolist())
                gts[patient].append(label.detach().cpu().tolist())

    return preds, gts

## Simple arithmetic mean
def pred_gt_by_patients(preds, gts):
    total_IDs = list(preds.keys())
    patient_preds, patient_gts = dict(), dict()
    for ID in total_IDs:
        ID_preds = preds[ID]
        ID_gts = gts[ID]
        patient_preds[ID] = np.max(ID_preds)
        patient_gts[ID] = np.mean(ID_gts)
    
    return patient_preds, patient_gts

def preds_and_gts(preds, gts):
    total_IDs = list(preds.keys())
    pred_list, gt_list = [], []
    for ID in total_IDs:
        pred_list.append(preds[ID])
        gt_list.append(gts[ID])
    return pred_list, gt_list

def ROC(preds, gts, SJS_size, CTR_size):
    auc = roc_auc_score(gts, preds)

    fpr, tpr, thresholds = roc_curve(gts, preds)
    J=tpr-fpr
    idx = np.argmax(J)

    best_thresh = thresholds[idx]
    sens, spec = tpr[idx], 1-fpr[idx]
    print(best_thresh)

    acc = (sens*SJS_size + spec*CTR_size) / (SJS_size+CTR_size)
    auc = roc_auc_score(gts, preds)

    plt.title("Roc Curve")
    plt.plot([0,1], [0,1], linestyle='--', markersize=0.01, color='black')
    plt.plot(fpr, tpr, marker='.', color='black', markersize=0.05)
    plt.scatter(fpr[idx], tpr[idx], marker='o', s=200, color='r',
                label = 'Sensitivity : %.3f (%d / %d), \nSpecificity = %.3f (%d / %d), \nAUC = %.3f , \nACC = %.3f (%d / %d)' % (sens, (sens*SJS_size), SJS_size, spec, (spec*CTR_size), CTR_size, auc, acc, sens*SJS_size+spec*CTR_size, SJS_size+CTR_size))
    plt.legend()
    plt.savefig(f"E:/Results/SJS/Figures/{control_group}_SJS(max_vote).png")
    
    return