import matplotlib.pyplot as plt

def plot(loss_hist, acc_hist):
    plt.figure(figsize=(20,10))
    plt.subplot(121), plt.plot(loss_hist, label='train_loss')
    plt.title('Train Loss')

    plt.subplot(122), plt.plot(acc_hist, label='train_acc')
    plt.title('Train Acc')
    plt.savefig("./Figures/Train.png")