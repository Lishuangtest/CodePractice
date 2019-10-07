import os
import torch
import torch.nn as nn      #就用nn来import各种的层了
import torch.utils.data as Data
import torchvision       #torchvision包含了很多数据库，其中就包含了图片的数据库
import matplotlib.pyplot as plt

#比如说我们要进行一个EPOCH的计算
EPOCH = 1
BATCH_SIZE = 50  #即批处理参数，即每一批处理多少的样本数量
LR = 0.001       #学习效率
DOWNLOAD_MNIST = False  #还没下载好mnist数据集，就设置为True


#那我们的mnist数据怎么下载呢？我们通过下面的方式跑去它的网站下载mnist
train_data=torchvision.datasets.MNIST(
    root='./mnist',         #将下载下来的mnist数据保存在mnist的一个文件夹中,下载好后就保存在里面
    train=True,             #True会给你下载Training-data数据集的数据点，否则会给你Text-data的数据集的数据点。
    transform=torchvision.transforms.ToTensor(),  #把下载的数据改成Tensor的格式，附在mnist文件夹里.我们把数据的值变为（0,1）之间，我们这个数据是在一张灰度图上，灰度图上的灰度为0-255，这便是将灰度图片每一个像素点的值压缩到0-1之间（即每一个像素点的值可能为0-255之间的一个，我们把它压缩到0-1之间）
    download=DOWNLOAD_MNIST  #这边就是下载数据集了
)
#为了呈现出Training-data的样子，这边做了个画图 的功能
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')   #将Training-Data的第一张图片呈现出来
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

#现在我们有了training data，那我们是不是要开始用data loader了呢？怎么创建呢？（我们这边要用Training_data来用作分批次训练数据）
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
#我们现在要有一个测试集，因为我们已经有了一个训练集（training data），并且用train_loader对训练集进行分批训练了
test_data=torchvision.datasets.MNIST(root='./mnist',train=False)  #我们这边提取出来的不是training_data，而是test-data，因为train=False
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # unsqueeze是为了将BATCH-SIDE的维度加上去,后面除以一个255，是因为text_data数据还是0-255之间的，我们要把text_data的数据映射到0-1之间。这边的：2000是只取这个数据集上前2000个数字
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):      #开始建立我们的CNN网络
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(                #图片为（1,28,28）的维度，即高度为1（只有1张图片），长和宽为28
                in_channels=1,        #说明这张图片有多少高度，比如RGB图片高度为3，有三个层；又如灰度图片它只有1个层。我们这边为灰度图片，你刚才也看过，只有1层。
                out_channels=16,      #这里为过滤器个数，之前讲过，这里就不讲了。
                kernel_size=5,        #代表图片过滤器的长和宽都为5
                stride=1,             #过滤器步长为1
                padding=2,            #图片周围铺两层0，为了适应过滤器的长和宽以及步长。顺便说一下，像素点的数值是代表颜色的，这边的灰色图片，新加一圈像素点为0，表示黑色，像素点为255，表示白色。这边如果要你通过过滤器过滤后生成的图片和原本的图片尺寸一致。假如stride=1,那么padding要等于padding=(kernel_size-1)/2,比如这边kernel_size=5,那么不管原尺寸照片大小，只要padding=（5-1）/2=2,那么由过滤器生成的图片尺寸就和原图片尺寸一致
            ),      #卷积层，一个过滤器，英文叫fielter,就是扫描图片的那个过滤器，这个过滤器有长度，宽度，高度，是一个3维的过滤器，官方叫卷积核，之前有看过，这里不深入了解了。
                    #这边的话就变为维度为（16,28,28），即高度为16(有16张图片)，图片长和宽为28
            nn.ReLU(),        #神经网络，非线性激活层，就是用激活函数来激活上面传下来的数据
            nn.MaxPool2d(kernel_size=2),   #最大池化层,就是做的是筛选重要的部分往下传,这边kernel_size=2,就是选择2*2的过滤器，在这个区域选择最大的那个值。这边最大池化层操作时，我们让上一层的卷积层长宽更窄，但是高度不变
                                           #经过最大池化层，图片高度不变，但是经过2*2矩阵中只取一个值，图片的长和宽只剩下一半了，图片的长和宽只剩下14*14了，图片的维度为（16,14,14）了
        )      #这就是一层卷积层
        self.conv2=nn.Sequential(  #图片的维度为（16,14,14）
            nn.Conv2d(16,32,5,1,2),    #这边的参数和上面的参数相对应，只不过我们没有写卷积层中的参数并赋值了。要记住这个参数顺序，要是搞错了，会很麻烦。
            #这边图片的维度又变为（32,14,14）了
            nn.ReLU(),
            nn.MaxPool2d(2),         #这边pooling有两种，一种是maxpooling,一种是avgpooling,一般我们用的话使用maxpooling来做。这边的参数也和上一层中的参数相对应
            #经过这边的池化层，我们图片的维度变为（32,7,7）了
        )
        #输出层
        self.out=nn.Linear(32*7*7,10)    #因为图片中包含0-9中不同的数字，所以我们要将图片分为10类，每个数字为一类。
        #但是在完全连接层中，我们要将三维的数据展平为二维的数据，展平的过程如下所示
    def forward(self, x):
        x=self.conv1(x)      #我们将输入的图片x输入到第一层卷积层中，得到结果x
        x=self.conv2(x)      #将第一层卷积层得到的结果x输入到第二层卷积层中
                             #这边要考虑batch，即要将batch也要加入维度中，这边的维度为（batch,32,7,7）
        x=x.view(x.size(0),-1)         #这边我们是将三维向量进行一个展平的操作
                                       #前面的x.size(0)就是将batch的维度保留，后面-1的过程就是将（32,7,7）（由一张图片演化来的）全部变为一起，变为一个向量
        output=self.out(x)   #现在我们就可以将self.out(x)函数接受过来了，即接受32*7*7的输入和10个类别的输出
        return output, x
cnn=CNN()
print(cnn)
#下面这是训练的过程
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()    #这边续联的过程用交叉熵损失函数
#这下面运用上面两行的准备工作，开始正式训练，和我们以前讲的过程一样的
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:              #每50步看一下我们的训练效果
            test_output, last_layer = cnn(test_x)     #我们每50步查看一下在text_data中有多少图片预测对了
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

test_output, _ = cnn(test_x[:10])      #将test_data前10个真实值放入test_out中
pred_y = torch.max(test_output, 1)[1].data.numpy()    #将tese_data前10个预测值输入pred_y中
print(pred_y, 'prediction number')       #将前10个预测值输出
print(test_y[:10].numpy(), 'real number')  #将对应的真实数据输出，看下有没有对应上




