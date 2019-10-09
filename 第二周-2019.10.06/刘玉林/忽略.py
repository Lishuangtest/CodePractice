import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

#超参数
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(32, 1)
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        #x  (batch, time_step, input_size)
        #h_state (n_layers, batch, hidden_size)
        #r_out (batah, time_step, hidden_size)
        outs = []    #创建一个列表
        for time_step in range(r_out.size(1)):    #size(1)表示r_out第二维度的大小
            outs.append(self.out(r_out[:, time_step, :]))   #时间点上的输入，输出存在list中
            return torch.stack(outs, dim=1),h_state    #输出为list形式，将其变为tensor形式

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None

for step in range(60):
    start, end = step*np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))   #shape (batch, time_step, input_size)

    prediction, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data)    #重新将数据包进Variable

    loss = loss_func(prediction, y)
    optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 优化参数
