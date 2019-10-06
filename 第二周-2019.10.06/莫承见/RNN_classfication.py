import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#超参数
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28            #RNN时间步代表图片的高度
INPUT_SIZE = 28           #RNN输入大小代表图片宽度
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)    #每一次一批一批的训练数据比较有效

test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,      #一般顺序为（time_step, batch, input）,此语句把batch放到第一位置
        )
        self.out = nn.Linear(64,10)     #输出为10个类代表10个数字

    def forward(self,x):
        r_out, (h_n, h_c) = self.rnn(x, None)   #r_out存储了28个输出
        out = self.out(r_out[:, -1, :])         #取r_out的最后一步输出[batch, time_step, input]
        return out
rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # 优化所有cnn的参数
loss_func = nn.CrossEntropyLoss()                       # 使用交叉熵损失，因为我们使用的是label的数据

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):    #将batch data返回出来
        b_x = Variable(x.view(-1, 28, 28))          #将x reshape to (batcn的第一维度, time_step, input_size)
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 优化每一步

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = sum(pred_y == test_y)/test_y.size
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
