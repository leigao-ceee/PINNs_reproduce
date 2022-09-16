import numpy as np
import matplotlib.pyplot as plt
from visual_data import matplotlib_vision
from gPINN_forward import Net, inference, get_u
import torch
import os
pi = np.pi

if __name__ == '__main__':
    name = '1D Poisson'
    work_path = os.path.join('work', name)
    result = torch.load(os.path.join(work_path, 'latest_model.pth'))
    log_loss = result['log_loss']

    Visual = matplotlib_vision('/', input_name=('x', 'y'), field_name=('f',))
    plt.figure(2, figsize=(10, 5))
    plt.clf()
    Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
    Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'bcs_loss')
    Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
    plt.title('training loss')

    nodes = np.linspace(0, pi, 101, dtype=np.float32)[:, None]
    field = get_u(nodes)
    bounds_ind = [0, 100]
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    nodes_train = torch.tensor(nodes, dtype=torch.float32).to(device)
    field_train = torch.tensor(field, dtype=torch.float32).to(device)
    Net_model = Net(planes=[1] + [20] * 3 + [1], ).to(device)
    Net_model.load_state_dict(torch.load(os.path.join(work_path, 'latest_model.pth'))['model'])
    field_pred, equation = inference(nodes_train, Net_model)
    plt.figure(3, figsize=(18, 5))
    plt.clf()
    plt.subplot(121)
    Visual.plot_value(nodes_train.detach().cpu().numpy(), field_train.cpu().numpy(), 'true')
    Visual.plot_value(nodes_train.detach().cpu().numpy(), field_pred.cpu().detach().numpy(), 'pred')
    plt.title('results comparison')

    plt.subplot(122)
    Visual.plot_value(nodes_train.detach().cpu().numpy(), equation.detach().cpu().numpy(), 'equation')
    plt.title('equation residual')
    plt.show()