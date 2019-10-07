import torch
import numpy as np

data = [[1,2],[3,4]]
data1=[1,2,3,4]
tensor1 = torch.FloatTensor(data1)
tensor = torch.FloatTensor(data)    # 32bit floating point
data = np.array(data)
print(
    '\nnumpy ', data.dot(data),
    '\ntorch ', tensor1.dot(tensor1)
)

