import torch
from torch.utils.data import DataLoader
from basic_setup import device, hyperparameters, model_save_path, train_orientation
from Simple_Model import ASD_Model
from CTRL_Aug_Dataset import test_dataset
from Test_plot import plot

model = ASD_Model().to(device)

test_loader = DataLoader(test_dataset, batch_size=hyperparameters["BATCH_SIZE"], shuffle=True)

def test_loop(dataloader, model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pred=[]
    target=[]

    for (X,y) in dataloader:
        for t in y:
            target.append(t[1].detach().tolist())

        X = X.to(device).float()
        y = y.to(device).float()

        output = model(X)

        for o in output:
            print(f"Output: {o}")
            pred.append(o[1].detach().cpu().tolist())

    return target, pred

target, pred = test_loop(test_loader, model, model_save_path)

plot(target, pred)