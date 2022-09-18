import numpy as np
import torch
import torch.nn as nn
from basic_model import DeepModel_single, gradients
import time
from tqdm import trange
import matplotlib.pyplot as plt
import os
pi = np.pi

name = '1D Poisson'
work_path = os.path.join('work', name)
isCreated = os.path.exists(work_path)
if not isCreated:
    os.makedirs(work_path)


# visualize function and generate data
def get_u(x):
    sol = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        sol += 1 / i * np.sin(i * x)
    return sol


nodes = np.linspace(0, pi, 101, dtype=np.float32)[:, None]
field = get_u(nodes)
bounds_ind = [0, 100]
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
nodes_train = torch.tensor(nodes, dtype=torch.float32).to(device)
field_train = torch.tensor(field, dtype=torch.float32).to(device)
# plt.figure(figsize=(12, 10))
# plt.plot(nodes, field)
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.show()


# network and
class Net(DeepModel_single):
    def __init__(self, planes):
        super(Net, self).__init__(planes, active=nn.Tanh())

    def equation(self, inn_var, out_var):
        dudx = gradients(out_var, inn_var)
        d2udx2 = gradients(dudx, inn_var)
        d3udx3 = gradients(d2udx2, inn_var)
        f = 8 * torch.sin(8*inn_var)
        for i in range(1, 5):
            f += i * torch.sin(i * inn_var)
        pde = -d2udx2 - f
        dfdx = torch.cos(inn_var) + 4 * torch.cos(2 * inn_var) + 9 * torch.cos(3 * inn_var) \
            + 16 * torch.cos(4 * inn_var) + 64 * torch.cos(8 * inn_var)
        gpde = - d3udx3 - dfdx
        return pde, gpde


def output_transform(x, y):
    # y is the output from NN
    return x + torch.tanh(x) * torch.tanh(np.pi - x) * y


def train(inn_var, bounds, out_true, model, Loss, optimizer, scheduler, log_loss):
    def closure():
        optimizer.zero_grad()
        out_var = model(inn_var)
        res_pde, res_gpde = model.equation(inn_var, out_var)
        eqs_loss = Loss(res_pde, torch.zeros_like(res_pde, dtype=torch.float32))
        bcs_loss = Loss(out_var[bounds], out_true[bounds])
        loss_batch = bcs_loss + eqs_loss
        loss_batch.backward()
        data_loss = Loss(out_var, out_true)
        log_loss.append([eqs_loss.item(), bcs_loss.item(), data_loss.item()])
        return loss_batch
    optimizer.step(closure)
    scheduler.step()


def inference(inn_var, model):
    inn_var.requires_grad_(True)
    out_pred = model(inn_var)
    res_pde, res_gpde = model.equation(inn_var, out_pred)

    return out_pred, res_pde


if __name__ == '__main__':
    # 建立网络
    Net_model = Net(planes=[1] + [20] * 3 + [1], ).to(device)
    # 损失函数
    L2loss = nn.MSELoss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=0.001, betas=(0.7, 0.9))
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=[50000, 70000, 80000], gamma=0.1)
    # 可视化

    star_time = time.time()
    log_loss = []
    pbar = trange(80000)

    inn_var = nodes_train
    inn_var.requires_grad_(True)

    # Training
    for iter in pbar:
        learning_rate = Optimizer.state_dict()['param_groups'][0]['lr']
        train(inn_var, bounds_ind, field_train, Net_model, L2loss, Optimizer, Scheduler, log_loss)

        # if iter > 0 and iter % 200 == 0:
        # print('iter: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, dat_loss: {:.3e}, bon_loss1: {:.3e}, cost: {:.2f}'.
        #       format(iter, learning_rate, log_loss[-1][0], log_loss[-1][-1], log_loss[-1][1], time.time()-star_time))

        pbar.set_postfix({'lr': learning_rate, 'dat_loss': log_loss[-1][-1], 'cost:': time.time() - star_time,
                          'eqs_loss': log_loss[-1][0], 'bcs_loss': log_loss[-1][1], })
    torch.save({'log_loss': log_loss, 'model': Net_model.state_dict(), }, os.path.join(work_path, 'latest_model.pth'))