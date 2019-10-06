import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
torch.manual_seed(2)
EPOCH=1
BATCH_SIZE=64
TIME_STEP=28
INPUT_SIZE=28
LR=0.01
DOWNLOAD_MNIST=False
train_data=torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
test_data=torchvision.datasets.MNIST(root='./mnist',train=False)
train_loader=Data.DataLoader(dataset=train_data,shuffle=True,batch_size=BATCH_SIZE)
test_x=test_data.data.type(torch.FloatTensor)[:2000]/255
test_y=test_data.targets.numpy().squeeze()[:2000]
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Linear(64,10)
    def forward(self, x):
        r_out,(h_n,h_c)=self.rnn(x,None)
        out=self.out(r_out[:,-1,:])
        return out
rnn=RNN()
print(rnn)
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step,(x,b_y) in enumerate(train_loader):
            print(x)
            b_x=x.view(-1,28,28)
            print(b_x)
            output=rnn(b_x)
            # print(output,b_y)
            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')

