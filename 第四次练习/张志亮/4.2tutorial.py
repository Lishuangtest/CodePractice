import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

train_data = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(),download=DOWNLOAD_MNIST,)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]

class RNN(nn.Module):    # 定义RNN模块
    def __init__(self):  # 创建RNN中的函数，也叫方法
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(       # 如果用nn.RNN模块，它几乎不学习
            input_size=INPUT_SIZE,
            hidden_size=64,    # rnn的隐藏单元
            num_layers=1,      # rnn的层数
            batch_first=True,    # 输入和输出的批量大小为1s维度 列如
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)    # x (batch, time_step, input_step, input_size)
        out = self.out(r_out[:, -1, :])    # (batch, time step, input)
        return out

rnn = RNN()
print(rnn)