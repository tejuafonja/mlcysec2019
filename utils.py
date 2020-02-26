import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
def transform_adv_origin_pair_to_dataloader(data_pair, batch_size=32, shuffle=False):
    dataset = []
    for i in range(len(data_pair)):
        data, label = train_data[i]
        dataset.append([data, label])
    return DataLoader(dataset, batch_size, shuffle)


def plot_losses_for_train_test(plt, train_loss, val_loss, xtick_num=2):
    n_iter = len(train_loss)
    iterations = [n for n in range(n_iter)]
    plt.plot(iterations, train_loss, label='train loss')
    plt.plot(iterations, val_loss, label='val loss')
    plt.set_xlabel('iterations')
    plt.set_xticks([n for n in range(0, len(iterations), xtick_num)])
    plt.set_ylabel('losses')
    plt.legend()

def plot_accs_for_train_test(plt, train_acc, val_acc, xtick_num=2):
    n_iter = len(train_acc)
    iterations = [n for n in range(n_iter)]
    plt.plot(iterations, train_acc, label='train acc')
    plt.plot(iterations, val_acc, label='val acc')
    plt.set_xlabel('iterations')
    plt.set_xticks([n for n in range(0, len(iterations), xtick_num)])
    plt.set_ylabel('acc')
    plt.legend()