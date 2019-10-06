#%%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
EPOCH = 1
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),  # (0,1)
    download=DOWNLOAD_MNIST,
)

#%%
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap="gray")
plt.title("%i" % train_data.train_labels[0])
plt.show()


#%%
import torch.utils.data as Data

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(root="./mnist/", train=False)
test_x = (
    Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(
        torch.FloatTensor
    )[:2000]
    / 255.0
)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,  # 5*5的kernel
                stride=1,  # 每隔1步跳一个
                padding=2,  # padding=(kernel_size-1)/2=2
            ),  # (28,28,16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 选区域中最大的值作为筛选的特征 下采样器
            # (14,14,16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,  # 5*5的kernel
                stride=1,  # 每隔1步跳一个
                padding=2,  # padding=(kernel_size-1)/2=2
            ),  # -> (14,14,32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 选区域中最大的值作为筛选的特征 下采样器
            # -> (7,7,32)
        )
        # 32*7*7是(7,7,32)展平的结果
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch,32,7,7)
        x = x.view(x.size(0), -1)  # (batch,32*7*7)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

#%%
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            print(
                "EPOCH: ",
                epoch,
                "| train loss: %.4f" % loss.data,
            )


#%%
test_output = cnn(test_x[20:60])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, "prediction number")
print(test_y[20:60].numpy(), "real number")

"""
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
"""

#%%
