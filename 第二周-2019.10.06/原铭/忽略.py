import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
torch.manual_seed(2)
Epoch=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False
train_data=torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
plt.imshow(train_data.data[0].numpy(),cmap='gray')
plt.title('%i'%train_data.targets[0])
# plt.show()
test_data=torchvision.datasets.MNIST(root='./mnist',train=False)
# print(test_data.test_data.size())
# print(train_data.train_data.size())
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.targets[:2000]
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out=nn.Linear(32*7*7,10)
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return  output
cnn=CNN()
# print(cnn)
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
if __name__ == '__main__':
   for epoch in range(Epoch):
        for step,(b_x,b_y) in enumerate(train_loader):
            # print(b_x.size())
            # print(b_y[:10])
            output=cnn(b_x)
            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(output[:2],b_y[:2])
            # if step%50==0:
            #     test_output=cnn(test_x)
            #     pred_y=torch.max(test_output,1)[1].data.squeeze()
            #     accuracy=sum(pred_y==test_y)/test_y.size(0)
            #     print('%.4f'%loss.data,'accuracy:%.4f'%accuracy)
test_output=cnn(test_x[:10])
print(test_output)
print(test_output.size())
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y)
print(test_y[:10].data.numpy())




