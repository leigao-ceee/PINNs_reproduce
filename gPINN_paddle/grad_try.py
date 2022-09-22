import numpy as np
import paddle

paddle.enable_static()
paddle.incubate.autograd.enable_prim()


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.weight = self.create_parameter(shape=(2, 2), dtype='float32', is_bias=False)
        self.bias = self.create_parameter(shape=(2,), dtype='float32', is_bias=True)
        self.out = self.create_parameter(shape=(1,), dtype='float32', is_bias=True)
        self.add_parameter("weight", self.weight)
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        y = paddle.matmul(x, self.weight) * self.out + self.bias
        return paddle.tanh(y)


# 模拟一个形状变化输入数据
data = [
    np.random.rand(3, 2).astype(np.float32),
    np.random.rand(4, 2).astype(np.float32),
    np.random.rand(5, 2).astype(np.float32)
]

# 遍历输入数据，将形状作为 paddle.static.data 参数传入，此时形状是固定的
for i, input in enumerate(data):
    x = paddle.static.data(f'x{i}', shape=input.shape, dtype='float32')
    net = MyNet()
    y = net(x)

    grad1 = paddle.incubate.autograd.grad(y, x)
    grad2 = paddle.incubate.autograd.grad(grad1, x)
    loss = paddle.norm(grad2, p=2)

    opt = paddle.optimizer.Adam(0.01)
    opt.minimize(loss)

exe = paddle.static.Executor()
exe.run(paddle.static.default_startup_program())
print(net.parameters())
for epoch in range(10):
    loss_val, = exe.run(feed={f'x{i}': x for i, x in enumerate(data)}, fetch_list=[loss])
    print("epoch: ", epoch + 1, " loss: ", loss_val)
