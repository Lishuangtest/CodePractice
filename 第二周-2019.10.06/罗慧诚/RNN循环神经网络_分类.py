import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#创建一些超参数（Hyper Parameters）
EPOCH=1
BATCH_SIZE=64      #用于批训练的样本数量
TIME_STEP=28       #RNN考虑的多少时间点的数据
INPUT_SIZE=28      #每个时间点的数据含有多少数据点
                   #比如我们用mnist图片是28*28的长宽，那么TIME_STEP=28,表示这张图片的高为28，INPUT_SIZE=28,表示这张图片的长为28
LR=0.01                #学习率
DOWNLOAD_MNIST=True   #是否要下载MNIST，由于已经下好了MNIST，故不用

train_data=dsets.MNIST(root='./mnist',train=True,transform=transforms.ToTensor(),download=DOWNLOAD_MNIST) #这边要把mnist数据装成tensorflow类型的，这边download表示你要不要下载数据。如果已经download，就直接用。不然就下载数据
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)   #我们这边用Data_Loader来一批一批地训练网络
#test_data用于对模型的测试
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn=nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,    #隐藏层的神经元有多少个
            num_layers=1,      #RNN中就是相当于细胞结构，可能有一层细胞结构，多两层细胞结构等。多层细胞结构，自然更厉害，只不过RNN在计算时，拖的事件更长
            batch_first=True,  #一般输入数据时，RNN网络会有多个维度，比如（batch,time_step,input）这些维度。你将Batch放在第一个维度，那么你的batch_first才为True
        )    #这个就是我们使用LSTM的RNN体系
        self.out=nn.Linear(64,10)      #我们RNN输出的数据我们还要处理下，我们把上面处理过的数据接到一个完全连接层中，然后再输出结果

    def forward(self,x):
        r_out,(h_n,h_c)=self.rnn(x,None)      #一批的数据x传入到RNN中,数据的维度为（batch,time_step,input_size）
                                              #每次计算完一个时间的input_size后，RNN网络都会产生一个hidden_state.h_n,h_c分别为分线程的hidden_state和主线程的hidden_state.但是我们用不到最后一个时刻刻的hidden_state.我们要用的是out_put的最后时刻来进行分类的训练。这边的None代表我们首先得hidden_state有没有。没有为None.有的话即为hidden_state的那个数据了
        out=self.out(r_out[:,-1,:])  #这边的数据维度对应为：[batch,time step,input].这边是确定最终的输出（这边的r_out放了有从第一步到最后一步每一步的output,我们要选取的是它最后一个时间点的output。这边中间数取-1就是取最后一个output输出）
        return out

rnn=RNN()
print(rnn)

#接下来我们就开始进行训练的步骤
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)    #我们优化的是rnn的参数
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):    #整个数据训练完后，我们用train_loader把我们的Batch_side返回出来
        b_x = b_x.view(-1, 28, 28)                      #这边reshape一下，变成batch_side的第一个图片，即从batch_side的第一张图片开始训练
        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:      #每50步看一下训练误差的准确度
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        # 这便是看下前10个数据有没有预测对
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

