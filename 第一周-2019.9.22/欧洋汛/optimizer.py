#%%
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper parameter
LR = 0.01  # learning rate
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))
#%%
plt.scatter(x.numpy(), y.numpy())
plt.show()

#%%
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
#%%
#基本的net
class Net(torch.nn.Module):  # 继承自nn.Moudule模块
    def __init__(self):  # 搭建net
        super(Net, self).__init__()
        # 第一个参数为输入第二个为输出
        self.hidden = torch.nn.Linear(
            1, 20
        )  # 隐藏层信息 输入n_features输出n_hidden
        self.predict = torch.nn.Linear(20, 1)  # 输出为1

    def forward(self, x):
        # 进行反向传递
        x = F.relu(self.hidden(x))  # 把x输入到hidden层 输出给relu激活
        x = self.predict(x)  # 最后处理输出
        return x

#%%
# 建立4个不同的优化器下的网络
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
#net list
nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr = LR)
opt_Momentum = torch.optim.SGD(net_SGD.parameters(),lr=LR,momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimiezers = [opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]
loss_func = torch.nn.MSELoss()
losses_his = [[],[],[],[]]#record history
for epoch in range(EPOCH):
    print(epoch)
    for setp,(batch_x,batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net,opt,l_his in zip(nets ,optimiezers,losses_his):
            output = net(b_x)
            loss = loss_func(output,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data)  

#%%
import matplotlib.pyplot as plt
labels =['SGD','Momentum','RMSprop','Adam']
for i,l_his in enumerate(losses_his):
    plt.plot(l_his,label=labels[i])
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')
plt.ylim((0,0.2))
plt.show()

#%%
