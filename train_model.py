import os
from config import *
from Network.model import *
from Network.datamaker import *
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

argparse = argparse.ArgumentParser()
argparse.add_argument('--rho', type=float, default=0.8, help='dg / dr')
argparse.add_argument('--size', type=float, default=0.5, help='size of torus ,range: [0,1)')
argparse.add_argument('--run', type=int, default=0, help='repeat index')  # NEW


if __name__ == '__main__':
    args = argparse.parse_args()
    # folder
    currentFolder = os.path.dirname(os.path.abspath(__file__))
    modelFolder = os.path.join(currentFolder, 'data', 'saved-models')
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)
    figureFolder = os.path.join(currentFolder, 'data', 'figures')
    if not os.path.exists(figureFolder):
        os.makedirs(figureFolder)
    file_appendix = f"rho{args.rho}_size{args.size}_run{args.run}".replace('.', 'p')  # ←
    model_file = os.path.join(modelFolder, f"model_{file_appendix}.pth")
    log_file = os.path.join(modelFolder, f"log_{file_appendix}.npy")
    loss_file = os.path.join(figureFolder, f"loss_{file_appendix}.png")

    # generate data
    dataConfig = DatasetConfig()
    dataset = DatasetMakerRandom(dataConfig)

    # define model
    modelConfig = ModelConfig()
    modelConfig.rho = args.rho
    modelConfig.s0 = args.size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelConfig.device = device
    model = FFGC(modelConfig)

    # optimizer
    trainConfig = TrainConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=trainConfig.lr)
    # loop
    ks = []
    rs = []
    epoch_progress_bar = tqdm(range(trainConfig.epochs), desc='Epochs')
    for i in epoch_progress_bar:
        r, v = dataset.generate_data(samples=trainConfig.batch_size)
        loss = model.train_step(labels=r.to(device), vels=v.to(device), optimizer=optimizer)
        k, s = process_model_data(model, device)
        ks.append(k)
        rs.append(s)
        epoch_progress_bar.set_postfix({'loss': loss['isometry_loss'][-1]})
        epoch_progress_bar.update()

        if i % trainConfig.n_checkpoints == 0:
            torch.save(model.state_dict(), model_file)
            np.save(log_file, loss)
    epoch_progress_bar.close()
    print('finish training')

    # plot train result
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    fig_title = ["isometry loss", "size"]

    axs[0].plot(np.abs(loss["isometry_loss"]))
    axs[0].set_title(fig_title[0])

    axs[1].plot(loss["size"])
    axs[1].set_title(fig_title[1])

    fig.tight_layout()
    plt.savefig(loss_file)

    # plot cycle encoder
    n_module = 1
    ks = np.stack(ks).reshape(trainConfig.epochs, n_module, -1)
    rs = np.stack(rs).reshape(trainConfig.epochs, n_module, -1)
    colors = [
        'r-', 'g-', 'b-', 'y-', 'c-', 'm-', 'k-', 'w-', 'orange', 'purple',
        'pink', 'brown', 'gray', 'lime', 'olive', 'navy', 'teal', 'aqua', 'gold', 'indigo'
    ]
    labels = [f'k{i + 1}' for i in range(20)]
    fig = plt.figure(figsize=(8, 4*n_module))
    for m in range(n_module):
        ax1 = fig.add_subplot(n_module, 2, m * 2 + 1)

        for i in range(ks.shape[2]):
            plt.plot(ks[:, m, i], colors[i], label=labels[i])
        plt.legend()
        plt.ylabel('tori cycle encoder')
        plt.title(f'module {m + 1}')

        ax2 = fig.add_subplot(n_module, 2, m * 2 + 2)
        for i in range(rs.shape[2]):
            plt.plot(rs[:, m, i], colors[i], label=labels[i])
        plt.legend()
        plt.ylabel('resize tori cycle')
        plt.title(f'module {m + 1}')

    ks_file = os.path.join(figureFolder, f'{file_appendix}_ks.png')
    plt.savefig(ks_file)


