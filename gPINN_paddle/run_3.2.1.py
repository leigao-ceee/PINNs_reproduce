# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import argparse
import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn
import paddle.static as static

import visual_data
from basic_model_pdpd import DeepModel_single

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pi = np.pi


def get_args():
    parser = argparse.ArgumentParser('PINNs for Brinkman-Forchheimer model', add_help=False)
    parser.add_argument('-f', type=str, default="external")
    parser.add_argument('--net_type', default='gpinn', type=str)
    parser.add_argument('--epochs_adam', default=60000, type=int)
    parser.add_argument('--save_freq', default=2000, type=int, help="frequency to save model and image")
    parser.add_argument('--print_freq', default=200, type=int, help="frequency to print loss")
    parser.add_argument('--device', default=True, type=bool, help="use gpu")
    parser.add_argument('--work_name', default='Brinkman-Forchheimer-1', type=str, help="save_path")

    parser.add_argument('--Nx_EQs', default=15, type=int)
    parser.add_argument('--Nx_Sup', default=15, type=int)
    parser.add_argument('--Nx_Val', default=500, type=int)
    parser.add_argument('--g_weight', default=0.1, type=float)
    return parser.parse_args()


class Net_single(DeepModel_single):
    def __init__(self, planes, active):
        super(Net_single, self).__init__(planes, active=active, data_norm=[0, 0])
        self.v_e = self.create_parameter(shape=[1, ], dtype='float32', is_bias=False,
                                         default_initializer=nn.initializer.Constant(0.0))
        self.add_parameter("v_e", self.v_e)

    def out_transform(self, inn_var, out_var):
        return paddle.tanh(inn_var) * paddle.tanh(1 - inn_var) * out_var

    @property
    def get_parameter(self):
        return paddle.log(paddle.exp(self.v_e) + 1) * 0.1

    def equation(self, inn_var):
        out_var = self.forward(inn_var)
        out_var = self.out_transform(inn_var, out_var)
        dudx = paddle.incubate.autograd.grad(out_var, inn_var)
        d2udx2 = paddle.incubate.autograd.grad(dudx, inn_var)
        f = 8 * paddle.sin(8 * inn_var)
        for i in range(1, 5):
            f += i * paddle.sin(i * inn_var)
        eqs = -d2udx2 - f
        if opts.net_type == 'gpinn':
            d3udx3 = paddle.incubate.autograd.grad(d2udx2, inn_var)
            dfdx = paddle.cos(inn_var) + 4 * paddle.cos(2 * inn_var) + 9 * paddle.cos(3 * inn_var)
            + 16 * paddle.cos(4 * inn_var) + 64 * paddle.cos(8 * inn_var)
            g_eqs = - d3udx3 - dfdx
        else:
            g_eqs = paddle.zeros((1,), dtype=paddle.float32)
        return out_var, eqs, g_eqs


def build(opts, model):
    ## 采样

    # print(out_BCs)

    EQs_var = paddle.static.data('EQs_var', shape=[opts.Nx_EQs, 1], dtype='float32')
    EQs_var.stop_gradient = False
    # EQs_tar = paddle.static.data('EQs_tar', shape=[opts.Nx_EQs, 1], dtype='float32')

    # print(EQs_var)
    BCs_var = paddle.static.data('BCs_var', shape=[2, 1], dtype='float32')
    BCs_var.stop_gradient = True
    BCs_tar = paddle.static.data('BCs_tar', shape=[2, 1], dtype='float32')

    Sup_var = paddle.static.data('Sup_var', shape=[opts.Nx_Sup, 1], dtype='float32')
    Sup_var.stop_gradient = True
    Sup_tar = paddle.static.data('Sup_tar', shape=[opts.Nx_Sup, 1], dtype='float32')

    Val_var = paddle.static.data('Val_var', shape=[opts.Nx_Val, 1], dtype='float32')
    Val_var.stop_gradient = True
    Val_tar = paddle.static.data('Val_tar', shape=[opts.Nx_Val, 1], dtype='float32')

    eqs, g_eqs = model.equation(EQs_var)
    sup = model(Sup_var)
    sup = model.out_transform(Sup_var, sup)
    bcs = model(BCs_var)
    bcs = model.out_transform(BCs_var, bcs)
    val = model(Val_var)
    val = model.out_transform(Val_var, val)
    val_grad = paddle.incubate.autograd.grad(val, Val_var)

    # print(fields_all)

    EQsLoss = paddle.norm(eqs, p=2) ** 2 / opts.Nx_EQs  # 方程所有计算守恒残差点的损失，不参与训练
    gEQsLoss = paddle.norm(g_eqs, p=2) ** 2 / opts.Nx_EQs  # 方程所有计算守恒残差点的损失，不参与训练
    SupLoss = paddle.norm(Sup_tar - sup, p=2) ** 2 / opts.Nx_Sup  # 方程所有计算守恒残差点的损失，不参与训练
    BCsLoss = paddle.norm(BCs_tar - bcs, p=2) ** 2 / 2
    datLoss = paddle.norm(Val_tar - val, p=2) ** 2 / opts.Nx_Val  # 方程所有计算守恒残差点的损失，不参与训练

    total_loss = EQsLoss + SupLoss + gEQsLoss * opts.g_weight

    # print(model.parameters())
    # print(total_loss)
    optimizer = paddle.optimizer.Adam(0.001)
    optimizer.minimize(total_loss)
    # optimizer.minimize(EQsLoss)
    #
    return [val, val_grad], [EQsLoss, BCsLoss, SupLoss, gEQsLoss, datLoss, total_loss]


def sol(x):
    solution = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        solution += 1 / i * np.sin(i * x)
    return solution


def grad(x):
    solution = 1 + np.cos(8 * x)
    for i in range(1, 5):
        solution += np.cos(i * x)
    return solution


def gen_sup(num):
    xvals = np.linspace(1 / (num + 1), pi, num, dtype=np.float32, endpoint=False)
    yvals = sol(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def gen_all(num):
    xvals = np.linspace(0, pi, num, dtype=np.float32)
    yvals = sol(xvals)
    ygrad = grad(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1)), np.reshape(ygrad, (-1, 1))


if __name__ == '__main__':

    opts = get_args()
    print(opts)

    try:
        import paddle.fluid as fluid
        use_cuda = opts.device
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    except:
        place = None

    paddle.enable_static()

    # opts.Nx_EQs = N
    save_path = opts.net_type + '-Nx_EQs_' + str(opts.Nx_EQs)
    work_path = os.path.join('work', opts.work_name, save_path)
    tran_path = os.path.join('work', opts.work_name, save_path, 'train')
    isCreated = os.path.exists(tran_path)
    if not isCreated:
        os.makedirs(tran_path)

    # 将控制台的结果输出到a.log文件，可以改成a.txt
    sys.stdout = visual_data.Logger(os.path.join(work_path, 'train.log'), sys.stdout)

    sup_x, sup_u = gen_sup(opts.Nx_Sup)  # 生成监督测点
    train_x, train_u, _ = gen_all(opts.Nx_EQs)
    valid_x, valid_u, valid_g = gen_all(opts.Nx_Val)
    bcs_x, bcs_u = train_x[[0, -1]], train_u[[0, -1]]

    # print(bcs_x)

    paddle.incubate.autograd.enable_prim()

    planes = [1, ] + [20, ] * 3 + [1, ]
    Net_model = Net_single(planes=planes, active=nn.Tanh())
    [U_pred, U_grad], Loss = build(opts, Net_model)

    exe = static.Executor(place)
    exe.run(static.default_startup_program())
    prog = static.default_main_program()

    Visual = visual_data.matplotlib_vision('/', field_name=('u',), input_name=('x',))
    star_time = time.time()
    log_loss = []
    par_pred = []
    start_epoch = 0

    for epoch in range(start_epoch, 1+opts.epochs_adam):
        ## 采样

        exe.run(prog, feed={'EQs_var': train_x, 'BCs_var': bcs_x, 'BCs_tar': bcs_u, 'Sup_var': sup_x, 'Sup_tar': sup_u,
                            'Val_var': valid_x, 'Val_tar': valid_u}, fetch_list=[Loss[-1]])

        if epoch > 0 and epoch % opts.print_freq == 0:
            all_items = exe.run(prog, feed={'EQs_var': train_x, 'BCs_var': bcs_x, 'BCs_tar': bcs_u, 'Sup_var': sup_x,
                                            'Sup_tar': sup_u, 'Val_var': valid_x, 'Val_tar': valid_u},
                                fetch_list=[[U_pred, U_grad, Net_model.get_parameter] + Loss[:-1]])

            u_pred = all_items[0]
            u_grad = all_items[1]
            p_pred = all_items[2]
            loss_items = all_items[3:]

            log_loss.append(np.array(loss_items).squeeze())
            par_pred.append(np.array(p_pred))

            print('iter: {:6d}, lr: {:.1e}, cost: {:.2f}, val_loss: {:.2e}, v_e_pred: {:.2e}\n'
                  'EQs_loss: {:.2e}, BCS_loss: {:.2e}, Sup_loss: {:.2e}, Grad_loss: {:.2e}'.
                  format(epoch, 0.001, time.time() - star_time, float(loss_items[-1]), float(p_pred),
                         float(loss_items[0]), float(loss_items[1]), float(loss_items[2]), float(loss_items[3])))
            star_time = time.time()
            # print(np.array(par_pred).shape)
        if epoch > 0 and epoch % opts.save_freq == 0:
            plt.figure(1, figsize=(20, 10))
            plt.clf()
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            # Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'BCS_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'sup_loss')
            if opts.net_type == "gpinn":
                Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'grad_loss')
            plt.savefig(os.path.join(tran_path, 'log_loss.svg'))

            plt.figure(1, figsize=(20, 10))
            plt.clf()
            Visual.plot_loss(np.arange(len(par_pred)), np.array(par_pred)[:, -1], 'v_e_pred')
            Visual.plot_loss(np.arange(len(par_pred)), np.ones(len(par_pred)) * 0.001, 'EXACT')
            plt.xlabel("Epoch")
            plt.ylabel("v_e")
            # Visual.plot_loss(np.arange(len(par_pred)), np.array(par_pred)[:, 0], 'eqs_loss')
            plt.savefig(os.path.join(tran_path, 'par_pred.svg'))

            plt.figure(4, figsize=(20, 15))
            plt.clf()
            Visual.plot_value(valid_x, valid_u, 'EXACT')
            Visual.plot_value(valid_x, u_pred, opts.net_type)
            plt.plot(sup_x, sup_u, 'ko')
            plt.xlabel("x")
            plt.ylabel("u")
            plt.savefig(os.path.join(tran_path, 'pred_u.svg'))

            plt.figure(4, figsize=(20, 15))
            plt.clf()
            Visual.plot_value(valid_x, valid_g, 'EXACT')
            Visual.plot_value(valid_x, u_grad, opts.net_type)
            plt.xlabel("x")
            plt.ylabel("du/dx")
            plt.savefig(os.path.join(tran_path, 'grad_u.svg'))

            star_time = time.time()


            paddle.save({'log_loss': log_loss, 'par_pred': par_pred,
                         'valid_x': valid_x, 'valid_u': valid_u, 'valid_g': valid_g,
                         'u_pred': u_pred, 'u_grad': u_grad,
                         }, os.path.join(work_path, 'out_res.pth'), )
    paddle.save(prog.state_dict(), os.path.join(work_path, 'latest_model.pdparams'), )
    time.sleep(3)
    shutil.move(os.path.join(work_path, 'train.log'), tran_path)