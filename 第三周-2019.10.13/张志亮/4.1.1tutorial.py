import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# 超参数
EPOCH = 1           # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001          # leaning rate
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),    # (0, 1)
    download=True
)
# plot one example
# print(train_data.train_data.size())      # (60000, 28, 28)
# print(train_data.train_labels.size())    # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255. # shape
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2, # if stride = 1, padding = (kernel_size-1)/2= (5-1)/2
            ),  # -> (16, 28, 28)
            nn.ReLU(),   # -> (16, 28, 28)
            nn.MaxPool2d(kernel_size=2),   # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential( # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2), # -> (32, 14, 14)
            nn.ReLU(),  # ->(32, 14, 14)
            nn.MaxPool2d(2)  # ->(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)     # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)      # (batch, 32 * 7* 7)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)   # 网络结构
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)    # 优化所有CNN参数
loss_func = nn.CrossEntropyLoss()    # the target label is not one-hotted

    # training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   #gives batch data,normalize x when item
        b_x = Variable(x)    # batch x
        b_y = Variable(y)    # batch y
        output  = cnn(b_x)              # 输出卷积神经网络
        loss = loss_func(output, b_y)   # 交叉熵损失
        optimizer.zero_grad()           # 清算这个梯度
        loss.backward()                 # 反向传播，计算梯度
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum((pred_y == test_y) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            labels = test_y.numpy()[:plot_only]
            plot_with_labels(low_dim_embs, labels)
plt.ioff()
            # print 10 prediction from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')


