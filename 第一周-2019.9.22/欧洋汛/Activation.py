#%%
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)  # 得到一个torch
# print(x.size())
x = Variable(x)
x_np = x.data.numpy()

#%%
# 画图时候还是要转化为numpy
x_np = x.data.numpy()
xtensor = torch.FloatTensor(x_np)

y_relu = xtensor.relu().data.numpy()
y_sigmoid = xtensor.sigmoid().data.numpy()
y_tanh = xtensor.tanh().data.numpy()
#y_softmax= xtensor.softmax().data.numpy()

# plot now
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

# plt.subplot(224)
# plt.plot(x_np, y_soft, c='red', label='softplus')
# plt.ylim((-0.2, 6))
# plt.legend(loc='best')

plt.show()

#%%
