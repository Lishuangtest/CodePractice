#%%
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable

torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28  # rnn 时间步数 / 图片高度
INPUT_SIZE = 28  # rnn 每步输入值 / 图片每行像素
LR = 0.01  # learning rate
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle


# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root="./mnist/",  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True
)

test_data = torchvision.datasets.MNIST(root="./mnist/", train=False)
test_x = (
    Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000] / 255.0
)
test_y = test_data.test_labels.numpy().squeeze()[:2000]

#%%
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,  # 输入的大小
            hidden_size=64,  # 隐藏层节点个数
            num_layers=1,  # 隐藏层个数
            batch_first=True,  # (batch,time_step,input) 如果数据维度把batch放在前面就是True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # 这里反向传播中需要注意的是 (h_n,h_c)是通过rnn反向传播回来的
        # h是上一行的数据学习来的 也就是s(t) 将和r_out一起影响s(t+1)
        # 所有需要记录两个 一个是output 一个是学到的h
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])  # (batch,time_step,input)这里是选time_step为最后一步的输出
        # 只有最后一步是我们需要的
        return out


rnn = RNN()
print(rnn)


#%%
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵为把标签转化为One-hot编码

#%%
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))  # reshape x to (batch,time_step,input)
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size
            print(
                "Epoch: ",
                epoch,
                "train loss: %.4f" % loss.data,
                "test accuracy:%2f" % accuracy,
            )            


#%%
test_output = rnn(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10],'real number')

#%%
