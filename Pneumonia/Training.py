import time, torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Basic_settings import device, hyperparameters
from Dataset import train_dataset
from Model import PNEU_Model

train_loader = DataLoader(train_dataset, batch_size=hyperparameters["BATCH"], shuffle=True)

model = PNEU_Model().to(device)
loss = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["LR"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def train(dataloader, model, loss, optimizer, model_save_path):
    model.train()

    batches = len(dataloader)  # 64개 짜리 묶음이 몇 묶음 있는지
    datasize = len(dataloader.dataset)  # 전체 데이터의 개수가 얼마인지

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

            batch_loss = loss(output, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss_item += batch_loss.item()
            correct+=(output.argmax(1)==y.argmax(1)).detach().cpu().sum().item()

            if batch % 20 == 0:
                print(f"Batch loss : {(batch_loss/batches):>.5f} {batch}/{batches}")

        loss_hist.append(loss_item/batches)
        acc_hist.append(correct/datasize*100)

        print(f"Loss : {(loss_item/batches):>.5f} ACC : {(correct/datasize*100):>.2f}%")

        epoch_end = time.time()
        print(f"End epoch : {epoch+1}")
        print(f"Epoch time : {(epoch_end-epoch_start)//60} min {(epoch_end-epoch_start)%60} sec")
        print()

    torch.save(model.state_dict(), model_save_path)

    return loss_hist, acc_hist

loss_hist, acc_hist = train(train_loader, model, loss, optimizer, hyperparameters["model_save_path"])

plt.subplot(121), plt.plot(loss_hist, label='Train Loss')
plt.subplot(122), plt.plot(acc_hist, label="Train Accuracy")
plt.savefig("C:/Users/PC00/Desktop/HJ/AICOSS_2023(Pneumonia)/Results/Train.jpg")