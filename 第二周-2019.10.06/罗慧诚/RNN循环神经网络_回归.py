import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# # show data
# steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,   #我们是用sin值预测cos值，故输入输出均为1
            hidden_size=32,  # 就是我的hidden state有多少个神经元。每一个hidden均有这么多神经元
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)   #这32个神经元输入是接着我们某一个时间点的输入，接着一个输出y的坐标

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)      #x为当前的输入，h_state为上一个时间点的输出
                                                   #我们每一次都会输入h_state,所以我们新得到的h_state要作为下一次h_state的输入。注意，我们这边h_state是隐式传递的，不是显示的，故最后一步才有h_state显示的出来。而每一步的output均能输出
        outs=[]                                    #保留所有时间点的预测值，并且我们要想将3维的转换为2维的数据，我们要怎么加工呢？
        for time_step in range(r_out.size(1)):     #对于每一个时间点的数据，我们都要取出来，过hidden layer
            outs.append(self.out(r_out[:,time_step,:]))   #每一个时间点数据，我们都要取出，放在self.out中过一下，然后将最终的结果存进outs[]中
        return  torch.stack(outs,dim=1),h_state    #h_state的返回值我们要保存给下一次forward进来的新的CNN数据再用

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state=None
for step in range(60):          #我们只用了60次的数据，就可以拟合到很好的效果了
    start, end = step * np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])    #这便是将x和y包成variable的形式，这便是将x和y由1维数据变为3维数据 形式为（batch,time_step,input_size）
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction,h_state=rnn(x,h_state)
    h_state = h_state.data            #将这一次的h_state传给下一次的数据
    loss=loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw();
    plt.pause(0.05)

plt.ioff()
plt.show()