import torch
import numpy as np
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\n numpy array:', np_data,
    '\n torch_tensor:',  np_data,
    '\n tensor to array:', tensor2array,
)


