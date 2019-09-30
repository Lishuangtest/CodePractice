#%%
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# unsqueeze()用于升上到二维的tensor数据 torch只能处理二维情况
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # 加入噪音

x, y = Variable(x), Variable(y)
# scatter生成散点图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
#%%
# do some network
# 搭建了一个只有一层线性隐藏层的简单网络
class Net(torch.nn.Module):  # 继承自nn.Moudule模块
    def __init__(self, n_features, n_hidden, n_output):  # 搭建net
        super(Net, self).__init__()
        # 第一个参数为输入第二个为输出
        self.hidden = torch.nn.Linear(
            n_features, n_hidden
        )  # 隐藏层信息 输入n_features输出n_hidden
        self.predict = torch.nn.Linear(n_hidden, 1)  # 输出为1

    def forward(self, x):
        # 进行反向传递
        x = F.relu(self.hidden(x))  # 把x输入到hidden层 输出给relu激活
        x = self.predict(x)  # 最后处理输出
        return x


#%%
# 包含一个属性 10个神经元 输出一个值
net = Net(1, 10, 1)
# 查看网络
print(net)

#%%
# 加入optimizer优化器
# stochastic gradient descent最常用
# 这里使用的是随机梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=0.3)
loss_func = torch.nn.MSELoss()

plt.ion()  # 画图
plt.show()

for t in range(100):
    prediction = net(x)
    # 前面是预测 后面是真实值
    loss = loss_func(prediction, y)

    # 初始化操作
    optimizer.zero_grad()  # 把参数的梯度都设置为0
    loss.backward()  # loss反向传播 优化梯度
    optimizer.step()  # 利用优化器优化梯度
    if t % 10 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)
        plt.text(
            0.5,
            0,
            "Loss=%.4f" % loss.data.numpy(),
            fontdict={"size": 20, "color": "red"},
        )
        plt.pause(0.1)


#%%
# 保存模型
torch.manual_seed(1)  # reproducible

# 假数据
x = torch.unsqueeze(
    torch.linspace(-1, 1, 100), dim=1
)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)


def save():
    # 建网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    # 训练
    for _ in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 保存模型
    torch.save(net1, "net.pkl")  # entrie net save
    torch.save(net1.share_dict(), "net_params.pkl")  # 只保存参数

#提取模型
def restore_net():
    net2 = torch.load("net.pkl")


#只提取参数（更快）
def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )
    net3.load_state_dict('net_params.pkl')

