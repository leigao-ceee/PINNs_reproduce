import paddle
import paddle.nn as nn


def gradients(y, x, order=1, create=True):
    if order == 1:
        return paddle.autograd.grad(y, x, create_graph=create, retain_graph=True)[0]
    else:
        return paddle.stack([paddle.autograd.grad([y[:, i].sum()], [x], create_graph=True, retain_graph=True)[0]
                             for i in range(y.shape[1])], axis=-1)


class DeepModel_multi(nn.Layer):
    def __init__(self, planes, data_norm, active=nn.GELU()):
        super(DeepModel_multi, self).__init__()
        self.planes = planes
        self.active = active

        self.x_norm = data_norm[0]
        self.f_norm = data_norm[1]
        self.layers = nn.LayerList()

        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1], weight_attr=nn.initializer.XavierNormal()))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1, weight_attr=nn.initializer.XavierNormal()))
            self.layers.append(nn.Sequential(*layer))
            # self.layers[-1].apply(initialize_weights)

    def forward(self, in_var, in_norm=False, out_norm=False):
        if in_norm:
            in_var = self.x_norm.norm(in_var)
        # in_var = in_var * self.input_transform
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        if out_norm:
            return self.f_norm.back(paddle.concat(y, axis=-1))
        else:
            return paddle.concat(y, axis=-1)

    def loadmodel(self, File):
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at " + str(start_epoch))
            log_loss = checkpoint['log_loss']  # .tolist()
            return start_epoch, log_loss
        except:
            print("load model failed！ start a new model.")
            return 0, []

    def equation(self, inv_var, out_var):
        return 0


class DeepModel_single(nn.Layer):
    def __init__(self, planes, data_norm, active=nn.GELU()):
        super(DeepModel_single, self).__init__()
        self.planes = planes
        self.active = active

        self.x_norm = data_norm[0]
        self.f_norm = data_norm[1]
        self.layers = nn.LayerList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1], weight_attr=nn.initializer.XavierNormal()))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1], weight_attr=nn.initializer.XavierNormal()))

        self.layers = nn.Sequential(*self.layers)
        # self.apply(initialize_weights)

    def forward(self, inn_var, in_norm=False, out_norm=False):
        if in_norm:
            inn_var = self.x_norm.norm(inn_var)
        out_var = self.layers(inn_var)
        if out_norm:
            return self.f_norm.back(out_var)
        else:
            return out_var

    def loadmodel(self, File):
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at " + str(start_epoch))
            log_loss = checkpoint['log_loss']  # .tolist()
            return start_epoch, log_loss
        except:
            print("load model failed！ start a new model.")
            return 0, []

    def equation(self, **kwargs):
        return 0