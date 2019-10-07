import torch
import numpy as np

data = [[1,2],[3,4]]
tensor =torch.FloatTensor(data)    # 32bit floating point

print(
    '\nnumpy ', np.matmul(data,data),
    '\ntorch ', torch.mm(tensor,tensor)
)