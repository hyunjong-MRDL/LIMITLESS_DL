import PIL, torch, random, os
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

CFG={'SEED' : 42,  # 42~46
     'IMG_SIZE' : 256,
     'TEST_PORTION' : 0.5,  # Test set 비율
     'CONTROL' : "NORMAL",  # "NORMAL" or "NON_SJS"
     'save_path' : "E:/model_save_path/SJS/model_fusion/sample.pt",
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
    total_correct = dict()
    for epoch in range(CFG["EPOCHS"]):
        for patient in total_patients:
            pg_data, sg_data = [], []
            labels = y[patient]
            if patient in X_pg.keys():
                pg_data = X_pg[patient]
            if patient in X_sg.keys():
                sg_data = X_sg[patient]
            
            correct = 0
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

                    outputs = model(curr_pg, curr_sg)
                    loss = criterion(outputs, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (outputs > 0.5) and (label == 1): correct += 1
                    elif (outputs < 0.5) and (label == 0): correct += 1

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

                    outputs = model(curr_pg, curr_sg)
                    loss = criterion(outputs, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (outputs > 0.5) and (label == 1): correct += 1
                    elif (outputs < 0.5) and (label == 0): correct += 1
            
            print(f"Patient No. {patient} is {100 * (correct / len(labels)):.5f}% probable of SJS.")
            total_correct[patient] = (correct / len(labels))
        
        print(f"Epoch{epoch} finished.")

    torch.save(model.state_dict(), CFG["save_path"])
    return total_correct

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
                    preds[patient].append(output)
                    gts[patient].append(label)

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
                    preds[patient].append(output)
                    gts[patient].append(label)

    return preds, gts

def ROC(preds, gts, SJS_size, CTR_size):
    auc = roc_auc_score(gts, preds)

    fpr, tpr, thresholds = roc_curve(gts, preds)
    J=tpr-fpr
    idx = np.argmax(J)

    best_thresh = thresholds[idx]
    sens, spec = tpr[idx], 1-fpr[idx]
    print(best_thresh)

    acc = (sens*SJS_size + spec*CTR_size) / len(gts)
    auc = roc_auc_score(gts, preds)

    plt.title("Roc Curve")
    plt.plot([0,1], [0,1], linestyle='--', markersize=0.01, color='black')
    plt.plot(fpr, tpr, marker='.', color='black', markersize=0.05)
    plt.scatter(fpr[idx], tpr[idx], marker='o', s=200, color='r',
                label = 'Sensitivity : %.3f (%d / %d), \nSpecificity = %.3f (%d / %d), \nAUC = %.3f , \nACC = %.3f (%d / %d)' % (sens, (sens*SJS_size), SJS_size, spec, (spec*CTR_size), CTR_size, auc, acc, sens*SJS_size+spec*CTR_size, SJS_size+CTR_size))
    plt.legend()
    plt.savefig("E:Results/SJS/Figures/NOMRAL_SJS(vector_sum).png")
    
    return