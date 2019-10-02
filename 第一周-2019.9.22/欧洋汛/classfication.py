#%%
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 假数据
n_data = torch.ones(100, 2)  # 数据的基本形态
x0 = torch.normal(2 * n_data, 1)  # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2 * n_data, 1)  # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)  # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1)).type(torch.LongTensor)  # LongTensor = 64-bit integer

plt.scatter(
    x.data.numpy()[:, 0],
    x.data.numpy()[:, 1],
    c=y.data.numpy(),
    s=100,
    lw=0,
    cmap="RdYlGn",
)
plt.show()

#%%
# do some network
# todo: mothod1
# 搭建了一个只有一层线性隐藏层的简单网络
class Net(torch.nn.Module):  # 继承自nn.Moudule模块
    def __init__(self, n_features, n_hidden, n_output):  # 搭建net
        super(Net, self).__init__()
        # 第一个参数为输入第二个为输出
        self.hidden = torch.nn.Linear(
            n_features, n_hidden
        )  # 隐藏层信息 输入n_features输出n_hidden
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出为1

    def forward(self, x):
        # 进行反向传递
        x = F.relu(self.hidden(x))  # 把x输入到hidden层 输出给relu激活
        x = self.predict(x)  # 最后处理输出
        return x

#%%
# todo mothod2 快速搭建net
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 2)
)
print(net2)


#%%
# 包含2个属性 10个神经元 输出2个值
net1 = Net(n_features=2, n_hidden=10, n_output=2)
# 查看网络
print(net1)

#%%
# 加入optimizer优化器
# stochastic gradient descent最常用
# 这里使用的是随机梯度下降
optimizer = torch.optim.SGD(net1.parameters(), lr=0.1)
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵 概率和为1

plt.ion()  # 画图
plt.show()

for t in range(100):
    out = net1(x)
    # 前面是预测 后面是真实值
    loss = loss_func(out, y)

    # 初始化操作
    optimizer.zero_grad()  # 把参数的梯度都设置为0
    loss.backward()  # loss反向传播 优化梯度
    optimizer.step()  # 利用优化器优化梯度
    if t % 5 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(
            x.data.numpy()[:, 0],
            x.data.numpy()[:, 1],
            c=pred_y,
            s=100,
            lw=0,
            cmap="RdYlGn",
        )
        accuracy = sum(pred_y == target_y) / 200.0  # 预测中有多少和真实值一样
        plt.text(
            1.5, -4, "Accuracy=%.2f" % accuracy, fontdict={"size": 20, "color": "red"}
        )
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()
