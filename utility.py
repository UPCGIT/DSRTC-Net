import torch.nn as nn
import torch
import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from scipy.sparse import coo_matrix

matplotlib.rc("font", family='Microsoft YaHei')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Init_Weights_dsm(dsm,sigma_h,device):

    dsm = torch.tensor(dsm, dtype=torch.float32).flatten().to(device)
    N = dsm.shape[0]

    h_i = dsm.unsqueeze(0)
    h_j = dsm.unsqueeze(1)

    diff_squared = (h_i - h_j) ** 2
    sum_squared = (h_i + h_j) ** 2

    exp_component = -diff_squared / (sigma_h ** 2 * sum_squared)
    W = torch.exp(exp_component)

    Q = W.sum(dim=1, keepdim=True)
    W_global = W / Q

    return W_global

def weighted_difference_operator(W_global, H, W):

    N = H * W
    indices = torch.arange(N).reshape(H, W)

    sparse_matrices = []

    left_indices = indices[:, 1:]
    current_left_indices = indices[:, :-1]
    rows = current_left_indices.reshape(-1)
    cols = left_indices.reshape(-1)
    data = W_global[rows, cols]
    W_left = coo_matrix((data.cpu().numpy().astype(np.float64),
                         (rows.cpu().numpy(), cols.cpu().numpy())), shape=(N, N)).tocsr()
    sparse_matrices.append(W_left)

    right_indices = indices[:, :-1]
    current_right_indices = indices[:, 1:]
    rows = current_right_indices.reshape(-1)
    cols = right_indices.reshape(-1)
    data = W_global[rows, cols]
    W_right = coo_matrix((data.cpu().numpy().astype(np.float64),
                          (rows.cpu().numpy(), cols.cpu().numpy())), shape=(N, N)).tocsr()
    sparse_matrices.append(W_right)

    up_indices = indices[1:, :]
    current_up_indices = indices[:-1, :]
    rows = current_up_indices.reshape(-1)
    cols = up_indices.reshape(-1)
    data = W_global[rows, cols]
    W_up = coo_matrix((data.cpu().numpy().astype(np.float64),
                       (rows.cpu().numpy(), cols.cpu().numpy())), shape=(N, N)).tocsr()
    sparse_matrices.append(W_up)

    down_indices = indices[:-1, :]
    current_down_indices = indices[1:, :]
    rows = current_down_indices.reshape(-1)
    cols = down_indices.reshape(-1)
    data = W_global[rows, cols]
    W_down = coo_matrix((data.cpu().numpy().astype(np.float64),
                         (rows.cpu().numpy(), cols.cpu().numpy())), shape=(N, N)).tocsr()
    sparse_matrices.append(W_down)

    return sparse_matrices
def SAD(y_true, y_pred):

    y_true_norm = torch.nn.functional.normalize(y_true, dim=1)
    y_pred_norm = torch.nn.functional.normalize(y_pred, dim=1)

    cos_similarity = torch.sum(y_true_norm * y_pred_norm, dim=1)
    cos_similarity = torch.clamp(cos_similarity, -1.0, 1.0)

    sad = torch.acos(cos_similarity)

    return torch.mean(sad)

def MSE(y_true, y_pred):
    return torch.sqrt(torch.nn.functional.mse_loss(y_true, y_pred))


def Init_Weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'he':
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class MyTrainData(torch.utils.data.Dataset):
    def __init__(self, hsi_img, radar_img, transform=None):
        self.hsi_img = hsi_img
        self.radar_img = radar_img
        self.transform = transform

    def __getitem__(self, index):


        return self.hsi_img, self.radar_img

    def __len__(self):

        return 1




class HSI:
    def __init__(self, data, rows, cols,lidar_dims, gt, abundance_gt,DSM,MPN,M1,DSM_normalized):

        if data.shape[0] < data.shape[1]:
            data = data.transpose()


        self.bands = np.min(data.shape)
        self.cols = cols
        self.rows = rows
        self.lidar_dims = lidar_dims
        self.image = np.reshape(data, (self.rows, self.cols, self.bands))
        self.gt = gt
        self.abundance_gt = abundance_gt
        self.DSM = DSM
        self.MPN = MPN
        self.M1 = M1
        self.DSM_normalized = DSM_normalized

    def array(self):


        return np.reshape(self.image, (self.rows * self.cols, self.bands)),np.reshape(self.MPN, (self.rows * self.cols, self.lidar_dims))


def load_HSI(path):

    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, 'r')
    numpy_array = np.asarray(data['Y'], dtype=np.float32)
    numpy_array = numpy_array / np.max(numpy_array.flatten())

    n_rows = data['lines'].item()
    n_cols = data['cols'].item()
    lidar_dims = data['lidar_dims'].item()

    if 'GT' in data.keys():
        gt = np.asarray(data['GT'], dtype=np.float32)
    else:
        gt = None

    if 'S_GT' in data.keys():
        abundance_gt = np.asarray(data['S_GT'], dtype=np.float32)
    else:
        abundance_gt = None

    if 'DSM' in data.keys():
        DSM = np.asarray(data['DSM'], dtype=np.float32)
    else:
        DSM = None

    if 'DSM_normalized' in data.keys():
        DSM_normalized = np.asarray(data['DSM_normalized'], dtype=np.float32)
    else:
        DSM_normalized = None

    if 'MPN' in data.keys():
        MPN = np.asarray(data['MPN'], dtype=np.float32)
        MPN = MPN / np.max(MPN.flatten())
    else:
        MPN = None

    if 'M1'in data.keys():
        M1 = np.asarray(data['M1'], dtype=np.float32)
    else:
        M1 = None

    return HSI(numpy_array, n_rows, n_cols, lidar_dims, gt, abundance_gt,DSM,MPN,M1,DSM_normalized)

def pca(X, d):

    N = np.shape(X)[1]
    xMean = np.mean(X, axis=1, keepdims=True)
    XZeroMean = X - xMean
    [U, S, V] = np.linalg.svd((XZeroMean @ XZeroMean.T) / N)
    Ud = U[:, 0:d]
    return Ud

def hyperVca(M, q):

    L, N = np.shape(M)

    rMean = np.mean(M, axis=1, keepdims=True)
    RZeroMean = M - rMean
    U, S, V = np.linalg.svd(RZeroMean @ RZeroMean.T / N)
    Ud = U[:, 0:q]

    Rd = Ud.T @ RZeroMean
    P_R = np.sum(M ** 2) / N
    P_Rp = np.sum(Rd ** 2) / N + rMean.T @ rMean
    SNR = np.abs(10 * np.log10((P_Rp - (q / L) * P_R) / (P_R - P_Rp)))
    snrEstimate = SNR
    # print('SNR estimate [dB]: %.4f' % SNR[0, 0])
    SNRth = 18 + 10 * np.log(q)

    if SNR > SNRth:
        d = q
        U, S, V = np.linalg.svd(M @ M.T / N)
        Ud = U[:, 0:d]
        Xd = Ud.T @ M
        u = np.mean(Xd, axis=1, keepdims=True)
        Y = Xd / np.sum(Xd * u, axis=0, keepdims=True)

    else:
        d = q - 1
        r_bar = np.mean(M.T, axis=0, keepdims=True).T
        Ud = pca(M, d)

        R_zeroMean = M - r_bar
        Xd = Ud.T @ R_zeroMean
        c = [np.linalg.norm(Xd[:, j], ord=2) for j in range(N)]
        c = np.array(c)
        c = np.max(c, axis=0, keepdims=True) @ np.ones([1, N])
        Y = np.concatenate([Xd, c.reshape(1, -1)])
    e_u = np.zeros([q, 1])
    e_u[q - 1, 0] = 1
    A = np.zeros([q, q])
    A[:, 0] = e_u[0]
    I = np.eye(q)
    k = np.zeros([N, 1])

    indicies = np.zeros([q, 1])
    for i in range(q):
        w = np.random.random([q, 1])
        tmpNumerator = (I - A @ np.linalg.pinv(A)) @ w
        f = tmpNumerator / np.linalg.norm(tmpNumerator)
        v = f.T @ Y
        k = np.abs(v)
        k = np.argmax(k)
        A[:, i] = Y[:, k]
        indicies[i] = k

    indicies = indicies.astype('int')
    if (SNR > SNRth):
        U = Ud @ Xd[:, indicies.T[0]]
    else:
        U = Ud @ Xd[:, indicies.T[0]] + r_bar

    return U, indicies, snrEstimate

def numpy_RMSE(y_true, y_pred):

    num_cols = y_pred.shape[0]
    num_rows = y_pred.shape[1]
    diff = y_true - y_pred

    squared_diff = np.square(diff)
    mse = squared_diff.sum() / (num_rows * num_cols)
    rmse = np.sqrt(mse)
    return rmse


def order_abundance(abundance, abundanceGT):


    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_RMSE(a[:, :, i], agt[:, :, j])
    for i in range(0, num_endmembers):
        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance



def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos > 1.0: cos = 1.0
    return np.arccos(cos)



def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]

    SAD_matrix = np.zeros((num_endmembers, num_endmembers))
    SAD_index = np.zeros(num_endmembers).astype(int)
    SAD_endmember = np.zeros(num_endmembers)
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    e = endmembers.copy()
    egt = endmembersGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            SAD_matrix[i, j] = numpy_SAD(e[i, :], egt[j, :])
        SAD_index[i] = np.nanargmin(SAD_matrix[i, :])
        SAD_endmember[i] = np.nanmin(SAD_matrix[i, :])
        egt[SAD_index[i], :] = np.inf

    return SAD_index, SAD_endmember



def plotEndmembersAndGT(endmembers, endmembersGT, endmember_path, sadsave):
    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1
    SAD_index, SAD_endmember = order_endmembers(endmembersGT, endmembers)
    fig = plt.figure(num=1, figsize=(9, 9))
    plt.clf()
    title = "mSAD: " + np.array2string(SAD_endmember.mean(),
                                       formatter={'float_kind': lambda x: "%.3f" % x}) + " radians"
    plt.rcParams.update({'font.size': 15})
    st = plt.suptitle(title)
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[SAD_index[i], :], 'r', linewidth=1.0)
        plt.plot(endmembersGT[i, :], 'k', linewidth=1.0)
        ax.set_title(format(numpy_SAD(endmembers[SAD_index[i], :], endmembersGT[i, :]), '.3f'))
        ax.get_xaxis().set_visible(False)
        sadsave.append(numpy_SAD(endmembers[SAD_index[i], :], endmembersGT[i, :]))
    sadsave.append(SAD_endmember.mean())
    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.86)
    plt.savefig(endmember_path + '.png')


def order_abundance(abundance, abundanceGT):
    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_RMSE(a[:, :, i], agt[:, :, j])
    for i in range(0, num_endmembers):
        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance



def plotAbundancesSimple(abundances, abundanceGT, abundance_path, rmsesave):
    abundances = np.transpose(abundances, axes=[1, 0, 2])
    num_endmembers = abundances.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    abundance_index, MSE_abundance = order_abundance(abundanceGT, abundances)
    title = "RMSE: " + np.array2string(MSE_abundance.mean(),
                                       formatter={'float_kind': lambda x: "%.3f" % x})
    cmap = 'viridis'
    plt.figure(figsize=[10, 10])
    AA = np.sum(abundances, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[:, :, abundance_index[i]], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.set_title(format(numpy_RMSE(abundances[:, :, abundance_index[i]], abundanceGT[:, :, i]), '.3f'))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        rmsesave.append(numpy_RMSE(abundances[:, :, abundance_index[i]], abundanceGT[:, :, i]))

    rmsesave.append(MSE_abundance.mean())
    plt.tight_layout()
    plt.rcParams.update({'font.size': 15})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')



def plotAbundancesGT(abundanceGT, abundance_path):
    num_endmembers = abundanceGT.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0:
        n = n + 1
    title = '参照丰度图'
    cmap = 'viridis'
    plt.figure(figsize=[10, 10])

    AA = np.sum(abundanceGT, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundanceGT[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.rcParams.update({'font.size': 19})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')
    plt.draw()
    plt.pause(0.1)
    plt.close()
def plotAbundancesLIDAR(abundanceGT, abundance_path):
    num_endmembers = abundanceGT.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0:
        n = n + 1
    title = '参照丰度图'
    cmap = 'viridis'
    plt.figure(figsize=[10, 10])
    AA = np.sum(abundanceGT, axis=-1)

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundanceGT[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.rcParams.update({'font.size': 19})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')
    plt.draw()
    plt.pause(0.1)
    plt.close()
