from loss_function import *
from DSRTCNet import DSRTCNet
from utility import *
import torch
import os
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import time
import scipy.io as sio
import pandas as pd
import random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
datasetnames = {
                'Muufl': 'Muufl',
                'Houston':'Houston',
                'Trento_SNR30':'Trento_SNR30',
                'Trento_SNR20':'Trento_SNR20',
                'Trento_SNR10':'Trento_SNR10',
                }
dataset = 'Muufl'

hsi = load_HSI(
    "./Datasets/" + datasetnames[dataset] + ".mat"
)

data, lidar = hsi.array()
num_bands = data.shape[1]
num_endmembers = hsi.gt.shape[0]
num_pixels = data.shape[0]
num_cols = hsi.cols
num_rows = hsi.rows
lidar_dims = lidar.shape[1]
DSM_normalized = hsi.DSM_normalized
dsm = torch.tensor(DSM_normalized, dtype=torch.float32)
dsm = dsm.to(device)


end = []
abu = []
r = []

# 设置运行次数
num_runs = 1

# 设置输出路径和方法名称
output_path = './result'
method_name = 'DSRTCNet'

mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'

if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

if dataset == 'Muufl':
    learning_rate, epoch, batch_size = 0.005, 300,4
    alpha, beta = 0.2, 0.01
    lambda_reg = 0.001
    weight_decay_param = 5e-7
if dataset == 'Houston':
    learning_rate, epoch, batch_size = 0.001, 150, 1
    alpha, beta = 0.3, 0.2
    lambda_reg = 0.005
    weight_decay_param = 1e-7
if dataset == 'Trento_SNR10':
    learning_rate, epoch, batch_size = 0.001, 300, 1
    alpha, beta = 0.5, 0.1
    lambda_reg = 0.005
    weight_decay_param = 1e-7
if dataset == 'Trento_SNR20':
    learning_rate, epoch, batch_size = 0.001, 300, 16
    alpha, beta = 0.5, 0.1
    lambda_reg = 0.005
    weight_decay_param = 1e-7
if dataset == 'Trento_SNR30':
    learning_rate, epoch, batch_size = 0.001, 300, 16
    alpha, beta = 0.5, 0.1
    lambda_reg = 0.005
    weight_decay_param = 1e-7

for run in range(1, num_runs + 1):
    model = DSRTCNet(num_bands, num_endmembers, lidar_dims)
    model = model.to(device)

    M1_init = torch.from_numpy(hsi.M1).unsqueeze(2).unsqueeze(3).float()
    M1_init = M1_init.to(device)

    hsi_data = data.T
    lidar_data = lidar.T

    hsi_data = np.reshape(hsi_data, [num_bands, num_rows, num_cols])
    lidar_data = np.reshape(lidar_data, [lidar_dims, num_rows, num_cols])

    train_dataset = MyTrainData(hsi_img=hsi_data, radar_img=lidar_data, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    Init_Weights(model, 'xavier', 1)
    model_dict = model.state_dict()
    model_dict['decoder.0.weight'] = M1_init
    model.load_state_dict(model_dict)


    criterion = DSMEnhancedRegularizationLoss( lambda_reg=lambda_reg,alpha=0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_param)
    loss_func = nn.MSELoss(size_average=True, reduce=True, reduction='mean')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    apply_clamp_inst1 = NonZeroClipper()

    print('Start training!', 'run:', run)
    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name
    time_start = time.time()
    for epoch in range(epoch):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            model.train()
            abundance, endmembers, reconstructed = model(x, y)

            rmse_loss = RMSE(reconstructed, x)
            sad_loss = SAD(reconstructed, x)
            re_loss = alpha * sad_loss + beta * rmse_loss

            tv_loss = criterion(abundance, dsm)

            all_loss = re_loss  + tv_loss

            optimizer.zero_grad()
            all_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
            optimizer.step()
        if epoch % 100 == 0:
            print('Epoch:', epoch, '| rmse loss: %.4f' % rmse_loss.cpu().data.numpy(),
                  '| sad loss: %.4f' % sad_loss.cpu().data.numpy(),
                  '| DSE_loss: %.4f' % tv_loss.cpu().data.numpy(),
                  '| all loss: %.4f' % all_loss.cpu().data.numpy())

            scheduler.step()

        # 保存结果
    with torch.no_grad():
        model.eval()
        abundance, endmembers, reconstructed = model(x, y)

        endmembers = endmembers.cpu().numpy()
        endmembers = np.squeeze(endmembers)
        endmembers = endmembers.T

        abundance = abundance.cpu().numpy()
        abundance = np.squeeze(abundance)
        abundance = np.reshape(abundance, [num_endmembers, num_rows * num_cols])
        abundance = abundance.T
        abundance = np.reshape(abundance, [num_cols, num_rows, num_endmembers])

        reconstructed = reconstructed.cpu().numpy()
        reconstructed = np.squeeze(reconstructed)
        reconstructed = np.reshape(reconstructed, [num_bands, num_rows * num_cols])
        reconstructed = reconstructed.T

        plotEndmembersAndGT(endmembers, hsi.gt, endmember_path, end)

        plotAbundancesSimple(abundance, hsi.abundance_gt, abundance_path, abu)
        armse_y = np.sqrt(np.mean(np.mean((reconstructed - data) ** 2, axis=1)))
        r.append(armse_y)

        sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'EM': endmembers,
                                                                                  'A': abundance,
                                                                                  'Y_hat': reconstructed
                                                                                  })

    print('-' * 100)
    time_end = time.time()
end = np.reshape(end, (-1, num_endmembers + 1))
abu = np.reshape(abu, (-1, num_endmembers + 1))
pd.DataFrame(end).to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各端元SAD及mSAD运行结果.csv')
pd.DataFrame(abu).to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各丰度图RMSE及mRMSE运行结果.csv')
pd.DataFrame(r).to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '重构误差RE运行结果.csv')

plotAbundancesGT(hsi.abundance_gt, output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '参照丰度图')


print('程序运行时间为:', time_end - time_start)