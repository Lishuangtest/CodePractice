import numpy as np
import torch

#%%
np_data = np.arange(6).reshape((2, 3))  # 创建一个numpy
torch_data = torch.from_numpy(np_data)  # 创建一个torch_data
tensor2array = torch_data.numpy()  # 将torch转化为numpy
# print(
#     '\nnp_data\n',np_data,
#     '\ntorch_data\n',torch_data,
#     '\narray\n',tensor2array
# )

#%%
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32位float数据
# http://pytorch.org/docs/
# print("\nabs", "\nnumpy", np.abs(data), "\ntorch", torch.abs(tensor))
#%%
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
data = np.array(data)

print(
    "\nnumpy:",
    data.dot(data),
    "\ntorch:",
    torch.mm(tensor, tensor),  # mm为矩阵相乘
    # '\ntorch dot:',te.dot(tensor),#mm为矩阵相乘
)
#%%
from torch.autograd import Variable
import torch

# torch中的参数都是variable变量的方式存在的

tensor = torch.FloatTensor([[1, 2], [3, 4]])
# torch 常用的变量是占位符、常量、variable三种类型
# 新版本已经放弃使用 requires_grad参数 在torch1.0中已经默认可以追踪计算结果
variable = Variable(tensor, requires_grad=True)  # 设置为true即为追踪当前的变量

t_out = torch.mean(tensor * tensor)  # x^2
v_out = torch.mean(variable * variable)
print(tensor)  # tensor不可以方向传播
print(variable)  # Variable可以反向传播
#%%
v_out.backward()  # 误差的反向传递 计算的是v_out(计算的是MSE)
print(variable) #Variable类型
print(variable.data)#Variable.data是tensor类型
print(variable.numpy.numpy())#返回对应的numpy

#%%
