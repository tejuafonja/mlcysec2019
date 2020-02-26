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
    

def vis_predictions(adversarial_results, nonnan_idx=None, title='FGSM'):
    
    rows, cols = 10, 10
    data_num = len(adversarial_results)
    np.random.shuffle(adversarial_results)
    original_samples = np.zeros((data_num, 32, 32), )
    original_labels = np.zeros((data_num,),)
    perturbed_samples = np.zeros((data_num, 32, 32), )
    perturbed_labels = np.zeros((data_num, ), )
    
    for i, adv in enumerate(adversarial_results):
        original_labels[i] = adv[0]
        original_samples[i, :, :] = adv[1]
        perturbed_labels[i] = adv[2]
        perturbed_samples[i, :, :] = adv[3]
        
    if nonnan_idx is not None:
        original_labels = original_labels[nonnan_idx]
        original_samples = original_samples[nonnan_idx, :, :]
        perturbed_labels = perturbed_labels[nonnan_idx]
        perturbed_samples = perturbed_samples[nonnan_idx, :, :]
        
    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle("Original samples and {} samples".format(title), fontsize=24, y=1.04)
    
    for i in range(rows):
        selected_idx = np.where(original_labels == i)[0]
        img = original_samples[selected_idx][:5]
        adv_img = perturbed_samples[selected_idx][:5]
        img_label = original_labels[selected_idx][:5]
        adv_img_label = perturbed_labels[selected_idx][:5]
        
        for j in range(cols):
            k = i // 2
            
            if j % 2 == 0:
                ax[i][j].set_title('sample: {0}'.format(img_label[k]))
                ax[i][j].imshow(img[k])
                
            else:
                ax[i][j].set_title('adversarial: {}'.format(adv_img_label[k]))
                ax[i][j].imshow(adv_img[k])
                
            ax[i][j].axes.get_xaxis().set_visible(False)
            ax[i][j].axes.get_yaxis().set_visible(False)
            
    plt.tight_layout()
 
                                                    