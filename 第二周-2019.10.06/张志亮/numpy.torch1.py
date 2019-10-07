import torch
import numpy as np

# abs
data = [-1, -2, 1, 2]
tensor =torch.FloatTensor(data)    # 32bit

print(
    '\nabs',
    '\nnumpy ', np.abs(data),         # [1 2 1 2]
    '\ntorch ', torch.abs(tensor)      # [1 2 1 2]
)
