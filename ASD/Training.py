import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from basic_setup import hyperparameters, device, model_save_path, train_orientation
from Simple_Model import model
from Train_plot import plot
from CTRL_Aug_Dataset import train_dataset  # Control Group만 augmentation (Data imbalance, ASD가 Control의 약 5배)

if device == 'cuda':
    model = model.to(device)

loss = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["LR"], weight_decay=0.1)  # Weight decay (L2 regularization)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

train_loader = DataLoader(train_dataset, batch_size=hyperparameters["BATCH_SIZE"], shuffle=True)

def train_loop(dataloader, model, optimizer, loss_fn, model_save_path):
    model.train()

    size = len(dataloader)
    datasize = len(dataloader.dataset)

    loss_hist=[]
    acc_hist=[]

    for epoch in range(hyperparameters["EPOCHS"]):
        epoch_start = time.time()

        loss_item=0
        correct=0
        print(f"Start epoch : {epoch+1}")
        for batch, (X,y) in enumerate(dataloader):
            X = X.to(device).float()
            y = y.to(device).float()

            output = model(X)

            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item += loss.item()

            correct+=(output.argmax(1)==y.argmax(1)).detach().cpu().sum().item()

            if batch % 20 == 0:
                print(f"Batch loss : {(loss):>.5f} {batch}/{size}")

        scheduler.step()
        
        loss_hist.append(loss_item/size)
        acc_hist.append(correct/datasize*100)

        print(f"Loss : {(loss_item/size):>.5f} ACC : {(correct/datasize*100):>.2f}%")

        epoch_end = time.time()
        print(f"End epoch : {epoch+1}")
        print(f"Epoch time : {(epoch_end-epoch_start)//60} min {(epoch_end-epoch_start)%60} sec")
        print()

    torch.save(model.state_dict(), model_save_path)

    return loss_hist, acc_hist

loss_hist, acc_hist = train_loop(train_loader, model, optimizer, loss, model_save_path)

plot(loss_hist, acc_hist)