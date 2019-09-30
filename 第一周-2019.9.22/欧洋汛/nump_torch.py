#
import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))  # 创建一个numpy
torch_data = torch.from_numpy(np_data)  # 创建一个torch_data
tensor2array = torch_data.numpy()  # 将torch转化为numpy
# print(
#     '\nnp_data\n',np_data,
#     '\ntorch_data\n',torch_data,
#     '\narray\n',tensor2array
# )
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32位float数据
# http://pytorch.org/docs/
# print("\nabs", "\nnumpy", np.abs(data), "\ntorch", torch.abs(tensor))

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

