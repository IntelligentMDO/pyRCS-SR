
import os
from types import SimpleNamespace
from scipy.io import loadmat, savemat
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam, AdamW, RAdam
import numpy as np
import matplotlib.pyplot as plt
import time

filename_basic = 'zFlyingWing_0.3GHz_HH_2.30'

# 固定随机数
torch.manual_seed(0)
generator_tr = torch.Generator().manual_seed(0)

# 定义GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 加载数据
filename_load_mat = f'Dataset_NN_{filename_basic}.mat'
filename_save_mat = f'Output_NN_{filename_basic}.mat'
filename_save_pt = f'NN_{filename_basic}.pt'
filename_save_png_basic = f'NN_{filename_basic}'

def load_mat(filename_load_mat):
    data = SimpleNamespace()
    with h5py.File(filename_load_mat, 'r') as f:
        data_mat = f['/Dataset']

        data.psi_traintest = np.array(data_mat['Psi_TrainTest']).T
        data.index_train = np.array(data_mat['Index_Train']).squeeze() - 1
        data.index_test = np.array(data_mat['Index_Test']).squeeze() - 1
        data.am = np.array(data_mat['Am']).T
        data.phiu = np.array(data_mat['PhiU']).T
        data.coef_traintest = np.array(data_mat['Coef_TrainTest']).T

        data.maxfreq_fft = np.array(data_mat['MaxFreq_FFT']).squeeze()
        data.maxfreq_nyquist = np.array(data_mat['MaxFreq_Nyquist']).squeeze()
        data.range_freq = np.array(data_mat['Range_Freq']).squeeze()
        data.num_freq = np.array(data_mat['Num_Freq']).squeeze()

        data.am_inverse = np.array(data_mat['Am_Inverse']).T
        data.am_rbf = np.array(data_mat['Am_RBF']).T
    return data

data = load_mat(filename_load_mat)

# 归一化
def x_norm(x, min_x, range_x, mode):
    x_out = []
    if mode == 'norm':
        x_out = (x - min_x) / range_x
    elif mode == 'denorm':
        x_out = x * range_x + min_x
    return x_out

def y_norm(y, min_y, range_y, k_y, mode):
    y_out = []
    if mode == 'norm':
        y_out = ((y - min_y) / range_y * 2 - 1) * k_y
    elif mode == 'denorm':
        y_out = (y / k_y + 1) / 2 * range_y + min_y
    return y_out

min_x = np.min(data.psi_traintest[data.index_train], axis=0)
range_x = np.ptp(data.psi_traintest[data.index_train], axis=0)
data.x = x_norm(data.psi_traintest, min_x, range_x, mode='norm')

min_y = np.min(data.am, axis=0)
range_y = np.ptp(data.am, axis=0)
k_y = range_y / np.max(range_y, axis=0)
data.y = y_norm(data.am, min_y, range_y, k_y, mode='norm')

# 编码
def encode_fourier(x, freq):
    features = [x]
    for f in freq:
        features.append(np.sin(2 * np.pi * f * x))
        features.append(np.cos(2 * np.pi * f * x))
    return np.hstack(features)

freq_encode = np.linspace(data.range_freq[0], data.range_freq[1], data.num_freq)
data.x = encode_fourier(data.x, freq_encode)

# 定义参数
num_epoch = 200000

dim_in_fourier = data.num_freq * 2
dim_in_nl = 1
dim_h_nonlinear = [512, 512, 512]
dim_out = data.y.shape[1]
size_batch = int(np.round(data.index_train.shape[0] / 10))

k_fourier = 0.5

scale_nn = np.copy(k_y)

act_attn = 1  # 是否激活注意力
if act_attn == 1:
    lr_attn_init = 1e-4
else:
    lr_attn_init = 0
lr_nn_fourier_init = 1e-4
lr_nn_nonlinear_init = 2e-4
lr_min = 2e-6
lr_decayrate = 0.9

weightdecay_attn = 0
weightdecay_nn_fourier_init = 1
weightdecay_nn_fourier_final = 1e-4
weightdecay_nn_nonlinear = 1e-4

act_adoptlr = 1  # 是否激活学习率衰减
threshold_lr = 2e-2
patience_lr_init = 1000
patience_lr_final = 1000
cooldown_lr = 0

act_adoptstop = 1  # 是否激活早停,前提act_adoptlr=1
threshold_earlystop = 2e-2
patience_earlystop = 1000

num_epoch_plot = 20
index_am_plot = 1
index_bf_plot = 50000

# 转数据
data.x = torch.tensor(data.x, dtype=torch.float32).to(device)

min_y = torch.tensor(min_y, dtype=torch.float32).to(device)
range_y = torch.tensor(range_y, dtype=torch.float32).to(device)
k_y = torch.tensor(k_y, dtype=torch.float32).to(device)
data.y = torch.tensor(data.y, dtype=torch.float32).to(device)

data.am = torch.tensor(data.am, dtype=torch.float32).to(device)
data.phiu = torch.tensor(data.phiu, dtype=torch.float32).to(device)
coef_test = torch.tensor(data.coef_traintest[data.index_test], dtype=torch.float32).to(device)

am_inverse_test = torch.tensor(data.am_inverse[data.index_test], dtype=torch.float32).to(device)

# 定义抽样
class GenDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

def get_data(x, y, index):
    ds_tr = GenDataset(x[index], y)
    dl_tr = DataLoader(ds_tr, batch_size=size_batch, shuffle=True, generator=generator_tr)
    return dl_tr

dl_tr = get_data(data.x, data.y, data.index_train)

# 定义NN
class FCDNN(nn.Module):
    def __init__(self, dim_in_fourier, dim_in_nl, dim_h_nl, dim_out, k_fourier, scale_nn, act_attn):
        super().__init__()

        self.register_buffer('k_fourier', torch.tensor(k_fourier, dtype=torch.float32))
        if act_attn == 1:
            self.register_buffer('scale_nn_fourier', torch.tensor(1, dtype=torch.float32))
        else:
            self.register_buffer('scale_nn_fourier', torch.tensor(scale_nn, dtype=torch.float32))
        self.register_buffer('scale_nn_nl', torch.tensor(scale_nn, dtype=torch.float32))

        attn = nn.Parameter(torch.ones(dim_out, dim_in_fourier))
        self.attn = nn.ParameterList([attn])
        if act_attn == 1:
            nn.init.constant_(self.attn[0], 0)

        layer = [nn.Linear(dim_in_fourier, dim_out, bias=False)]
        self.layer_fourier = nn.Sequential(*layer)

        layer = []
        dim_prev = dim_in_nl
        for dim in dim_h_nl:
            layer.append(nn.Linear(dim_prev, dim))
            layer.append(nn.GELU())
            dim_prev = dim
        layer.append(nn.Linear(dim_prev, dim_out))
        self.layer_nonlinear = nn.Sequential(*layer)

    def forward(self, x):
        # (batch, 1, dim_in_fourier) * (1, dim_out, dim_in_fourier) = (batch, dim_out, dim_in_fourier)
        x_fourier = x[:, 1:].unsqueeze(1) * self.attn[0].unsqueeze(0)
        # ((batch, dim_out, dim_in_fourier) * (1, dim_out, dim_in_fourier)).sum(dim=2) = (batch, dim_out)
        y_fourier = (x_fourier * self.layer_fourier[0].weight.unsqueeze(0)).sum(dim=2) * self.k_fourier
        y_fourier = y_fourier * self.scale_nn_fourier

        y_nonlinear = self.layer_nonlinear(x[:, 0].reshape(-1, 1)) * (1 - self.k_fourier)
        y_nonlinear = y_nonlinear * self.scale_nn_nl
        return y_fourier + y_nonlinear, y_fourier, y_nonlinear

model_nn = FCDNN(dim_in_fourier=dim_in_fourier,
                 dim_in_nl=dim_in_nl,
                 dim_h_nl=dim_h_nonlinear,
                 dim_out=dim_out,
                 k_fourier=k_fourier,
                 scale_nn=scale_nn,
                 act_attn=act_attn).to(device)

optimizer = AdamW([{'params': model_nn.attn.parameters(),
                    'lr': lr_attn_init,
                    'weight_decay': weightdecay_attn},
                   {'params': model_nn.layer_fourier.parameters(),
                    'lr': lr_nn_fourier_init,
                    'weight_decay': weightdecay_nn_fourier_init},
                   {'params': model_nn.layer_nonlinear.parameters(),
                    'lr': lr_nn_nonlinear_init,
                    'weight_decay': weightdecay_nn_nonlinear}])

# 定义损失函数
func_loss = nn.MSELoss()

# 定义训练
def get_train(x, y, model_nn, func_loss, optimizer):
    model_nn.train()
    optimizer.zero_grad()
    y_pred, _, _ = model_nn(x)
    loss = func_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

strategy_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=lr_decayrate,
                                                         patience=patience_lr_init,
                                                         threshold=threshold_lr,
                                                         cooldown=cooldown_lr,
                                                         min_lr=lr_min,
                                                         threshold_mode='rel')

class EarlyStop:
    def __init__(self, patience, threshold):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif (self.best_loss - current_loss) / self.best_loss < self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0
        return self.early_stop

strategy_earlystop = EarlyStop(patience=patience_earlystop, threshold=threshold_earlystop)

# 训练
if os.path.exists(filename_save_pt):
    checkpoint = torch.load(filename_save_pt, weights_only=False)

    generator_tr.set_state(checkpoint['generator_tr'])

    model_nn.load_state_dict(checkpoint['statedict_model'])
    optimizer.load_state_dict(checkpoint['statedict_optimizer'])
    strategy_lr.load_state_dict(checkpoint['statedict_strategy_lr'])

    epoch_start, epoch_last = checkpoint['epoch_last'] + 1, checkpoint['epoch_last']

    loss_history_tr = checkpoint['loss_history_tr']
    loss_history_test = checkpoint['loss_history_test']
    loss_history_test_coef = checkpoint['loss_history_test_coef']

    lr_history_attn = checkpoint['lr_history_attn']
    lr_history_nn_fourier = checkpoint['lr_history_nn_fourier']
    lr_history_nn_nonlinear = checkpoint['lr_history_nn_nonlinear']

    weightdecay_history_attn = checkpoint['weightdecay_history_attn']
    weightdecay_history_nn_fourier = checkpoint['weightdecay_history_nn_fourier']
    weightdecay_history_nn_nonlinear = checkpoint['weightdecay_history_nn_nonlinear']

    state_earlystate = checkpoint['state_earlystate']
    time_cost = checkpoint['time_cost']

    for epoch in range(0, epoch_start):
        if epoch % num_epoch_plot == 0:
            print(f'{epoch + 1} | '
                  f'Tr:{loss_history_tr[epoch]:.1e} | '
                  f'Te:{loss_history_test[epoch]:.1e} | TeC:{loss_history_test_coef[epoch]:.1e} | '
                  f'L:{lr_history_nn_fourier[epoch]:.2e}')
else:
    epoch_start, epoch_last = 0, 0

    loss_history_tr = []
    loss_history_test = []
    loss_history_test_coef = []

    lr_history_attn = []
    lr_history_nn_fourier = []
    lr_history_nn_nonlinear = []

    weightdecay_history_attn = []
    weightdecay_history_nn_fourier = []
    weightdecay_history_nn_nonlinear = []

    state_earlystate = False

    time_cost = 0

time_start = time.time()

if epoch_start < num_epoch and not state_earlystate:
    for epoch in range(epoch_start, num_epoch):
        # 训练
        for x_batch, y_batch in dl_tr:
            get_train(x_batch, y_batch, model_nn, func_loss, optimizer)

        # 验证
        model_nn.eval()
        with torch.no_grad():
            am_pred, _, _ = model_nn(data.x[data.index_train])
        am_pred = y_norm(am_pred, min_y, range_y, k_y, mode='denorm')

        loss_tr = func_loss(am_pred, data.am).item()

        if epoch % num_epoch_plot == 0:
            model_nn.eval()
            with torch.no_grad():
                am_pred, _, _ = model_nn(data.x[data.index_test])
            am_pred = y_norm(am_pred, min_y, range_y, k_y, mode='denorm')
            coef_pred = torch.matmul(am_pred, data.phiu.T)

            loss_test = func_loss(am_pred, am_inverse_test).item()
            loss_test_coef = func_loss(coef_pred, coef_test).item()
        else:
            loss_test = loss_history_test[epoch - 1]
            loss_test_coef = loss_history_test_coef[epoch - 1]

        # 记录
        loss_history_tr.append(loss_tr)
        loss_history_test.append(loss_test)
        loss_history_test_coef.append(loss_test_coef)

        lr_history_attn.append(optimizer.param_groups[0]['lr'])
        lr_history_nn_fourier.append(optimizer.param_groups[1]['lr'])
        lr_history_nn_nonlinear.append(optimizer.param_groups[2]['lr'])

        weightdecay_history_attn.append(optimizer.param_groups[0]['weight_decay'])
        weightdecay_history_nn_fourier.append(optimizer.param_groups[1]['weight_decay'])
        weightdecay_history_nn_nonlinear.append(optimizer.param_groups[2]['weight_decay'])

        if epoch % num_epoch_plot == 0:
            print(f'{epoch + 1} | '
                  f'Tr:{loss_history_tr[epoch]:.1e} | '
                  f'Te:{loss_history_test[epoch]:.1e} | TeC:{loss_history_test_coef[epoch]:.1e} | '
                  f'L:{lr_history_nn_fourier[epoch]:.2e}')

        epoch_last = epoch

        # 判断早停
        if act_adoptstop == 1 and act_adoptlr == 1:
            if np.abs(lr_history_nn_fourier[epoch] - lr_min) < 1e-10:
                state_earlystate = strategy_earlystop(loss_history_tr[epoch])
                if state_earlystate and epoch % num_epoch_plot == 0:
                    break

        # 调整学习率
        if act_adoptlr == 1:
            strategy_lr.step(loss_history_tr[epoch])

        # 触发第一阶段调整
        if np.abs(optimizer.param_groups[2]['lr'] - lr_nn_nonlinear_init * lr_decayrate) < 1e-10:
            # 调整学习率
            optimizer.param_groups[0]['lr'] = lr_attn_init
            optimizer.param_groups[1]['lr'] = lr_nn_fourier_init
            optimizer.param_groups[2]['lr'] = lr_nn_fourier_init / 2
            strategy_lr.patience = patience_lr_final

            # 调整权重
            optimizer.param_groups[1]['weight_decay'] = weightdecay_nn_fourier_final

    time_cost_local = time.time() - time_start
    time_cost = time_cost + time_cost_local

    # 保存模型至.pt
    torch.save({'generator_tr': generator_tr.get_state(),

                'statedict_model': model_nn.state_dict(),
                'statedict_optimizer': optimizer.state_dict(),
                'statedict_strategy_lr': strategy_lr.state_dict(),

                'epoch_last': epoch_last,

                'loss_history_tr': loss_history_tr,
                'loss_history_test': loss_history_test,
                'loss_history_test_coef': loss_history_test_coef,

                'lr_history_attn': lr_history_attn,
                'lr_history_nn_fourier': lr_history_nn_fourier,
                'lr_history_nn_nonlinear': lr_history_nn_nonlinear,

                'weightdecay_history_attn': weightdecay_history_attn,
                'weightdecay_history_nn_fourier': weightdecay_history_nn_fourier,
                'weightdecay_history_nn_nonlinear': weightdecay_history_nn_nonlinear,

                'state_earlystate': state_earlystate,
                'time_cost': time_cost
                }, filename_save_pt)

else:
    time_cost_local = time.time() - time_start
    time_cost = time_cost + time_cost_local

print(f'{time_cost_local:.2f}')
print(f'{time_cost:.2f}')

# 转数据
data.x = data.x.cpu().detach().numpy()

min_y = min_y.cpu().detach().numpy()
range_y = range_y.cpu().detach().numpy()
k_y = k_y.cpu().detach().numpy()
data.y = data.y.cpu().detach().numpy()

data.am = data.am.cpu().detach().numpy()
data.phiu = data.phiu.cpu().detach().numpy()
coef_test = coef_test.cpu().detach().numpy()

am_inverse_test = am_inverse_test.cpu().detach().numpy()

model_nn = model_nn.to('cpu')

# 作图_1_损失曲线
fig = plt.figure(figsize=(10, 10), constrained_layout=True)

plt.subplot(221)
plt.plot(np.arange(len(loss_history_tr)) + 1, loss_history_tr)
plt.plot(np.arange(len(loss_history_test)) + 1, loss_history_test)
plt.yscale('log')
# plt.ylim(1e-10, 1e-3)
# plt.yticks([10 ** i for i in range(-10, -3)])
plt.text(len(loss_history_tr), loss_history_tr[0],
         f'{loss_history_tr[-1]:.1e}\n{loss_history_test[-1]:.1e}',
         ha='right', va='top')
plt.grid(True)
plt.title('Loss_TrTest')

plt.subplot(222)
plt.plot(np.arange(len(loss_history_test_coef)) + 1, loss_history_test_coef)
plt.yscale('log')
# plt.ylim(1e-13, 1e-6)
# plt.yticks([10 ** i for i in range(-13, -6)])
plt.text(len(loss_history_test_coef), loss_history_test_coef[0],
         f'{loss_history_test_coef[-1]:.1e}',
         ha='right', va='top')
plt.grid(True)
plt.title('Loss_Coef')

plt.subplot(223)
plt.plot(lr_history_attn)
plt.plot(lr_history_nn_fourier, '--')
plt.plot(lr_history_nn_nonlinear)
plt.yscale('log')
plt.grid(True)
plt.title('Lr')

plt.subplot(224)
plt.plot(weightdecay_history_attn)
plt.plot(weightdecay_history_nn_fourier)
plt.plot(weightdecay_history_nn_nonlinear)
plt.yscale('log')
plt.grid(True)
plt.title('WeightDecay')

plt.savefig(f'Loss_{filename_save_png_basic}.png')

# 作图_2_注意力*权重&注意力
plt.figure(figsize=(15, 10), constrained_layout=True)

plt.subplot(331)
attn = model_nn.attn[0].cpu().detach().numpy()[index_am_plot - 1, :]
weight_fourier = model_nn.layer_fourier[0].weight.cpu().detach().numpy()[index_am_plot - 1, :]

plt.plot(freq_encode, np.abs(attn * weight_fourier).reshape(data.num_freq, 2).sum(axis=1), '.')
plt.grid(True)
plt.title('Attn * Weight_Fourier')

plt.subplot(334)
attn = model_nn.attn[0].cpu().detach().numpy()[index_am_plot - 1, :]

plt.plot(freq_encode, np.abs(attn).reshape(data.num_freq, 2).sum(axis=1), '.')
plt.grid(True)
plt.title('Attn')

# 作图_2_Am
plt.subplot(332)
model_nn.eval()
with torch.no_grad():
    am_pred, _, _ = model_nn(torch.tensor(data.x))
am_pred = y_norm(am_pred.cpu().detach().numpy(), min_y, range_y, k_y, mode='denorm')

plt.plot(data.psi_traintest[data.index_train], data.am[:, index_am_plot - 1], 'k.')
plt.plot(data.psi_traintest, data.am_inverse[:, index_am_plot - 1], 'k--')
plt.plot(data.psi_traintest, data.am_rbf[:, index_am_plot - 1])
plt.plot(data.psi_traintest, am_pred[:, index_am_plot - 1])
plt.grid(True)
plt.title('Pred_Am')

plt.subplot(335)
model_nn.eval()
with torch.no_grad():
    am_pred, _, _ = model_nn(torch.tensor(data.x))
am_pred = y_norm(am_pred.cpu().detach().numpy(), min_y, range_y, k_y, mode='denorm')

plt.plot(data.psi_traintest,
         am_pred[:, index_am_plot - 1] - data.am_inverse[:, index_am_plot - 1])
plt.grid(True)
plt.title('Error_Am')

plt.subplot(338)
model_nn.eval()
with torch.no_grad():
    am_pred, am_pred_fourier, am_pred_nonlinear = model_nn(torch.tensor(data.x))
am_pred = am_pred.cpu().detach().numpy()
am_pred_fourier = am_pred_fourier.cpu().detach().numpy()
am_pred_nonlinear = am_pred_nonlinear.cpu().detach().numpy()

# plt.plot(data.psi_test, coef_pred)
plt.plot(data.psi_traintest, am_pred[:, index_am_plot - 1])
plt.plot(data.psi_traintest, am_pred_fourier[:, index_am_plot - 1])
plt.plot(data.psi_traintest, am_pred_nonlinear[:, index_am_plot - 1])
plt.grid(True)
plt.title('Output_Am_Layer')

# 作图_2_Coef
plt.subplot(333)
model_nn.eval()
with torch.no_grad():
    am_pred, _, _ = model_nn(torch.tensor(data.x))
am_pred = y_norm(am_pred.cpu().detach().numpy(), min_y, range_y, k_y, mode='denorm')

coef_rbf_plot = np.matmul(data.am_rbf, data.phiu[index_bf_plot - 1, :].T)
coef_pred_plot = np.matmul(am_pred, data.phiu[index_bf_plot - 1, :].T)

plt.plot(data.psi_traintest[data.index_train],
         data.coef_traintest[data.index_train, index_bf_plot - 1], 'k*')
plt.plot(data.psi_traintest,
         data.coef_traintest[:, index_bf_plot - 1], 'k--')
plt.plot(data.psi_traintest, coef_rbf_plot)
plt.plot(data.psi_traintest, coef_pred_plot)
plt.grid(True)
plt.title('Pred_Coef')

plt.subplot(336)
model_nn.eval()
with torch.no_grad():
    am_pred, _, _ = model_nn(torch.tensor(data.x))
am_pred = y_norm(am_pred.cpu().detach().numpy(), min_y, range_y, k_y, mode='denorm')

coef_pred_plot = np.matmul(am_pred, data.phiu[index_bf_plot - 1, :].T)

plt.plot(data.psi_traintest, coef_pred_plot - data.coef_traintest[:, index_bf_plot - 1])
plt.grid(True)
plt.title('Error_Coef')

plt.savefig(f'Pred_{filename_save_png_basic}.png')

# 作图_3_Am差异
plt.figure(figsize=(10, 5))

model_nn.eval()
with torch.no_grad():
    am_pred, _, _ = model_nn(torch.tensor(data.x))
am_pred = y_norm(am_pred.cpu().detach().numpy(), min_y, range_y, k_y, mode='denorm')

plt.plot(np.arange(data.am.shape[1]) + 1, np.mean((data.am_inverse - data.am_rbf) ** 2, axis=0))
plt.plot(np.arange(data.am.shape[1]) + 1, np.mean((data.am_inverse - am_pred) ** 2, axis=0))
plt.yscale('log')
plt.grid(True)

plt.savefig(f'AmDiff_{filename_save_png_basic}.png')

plt.show()








