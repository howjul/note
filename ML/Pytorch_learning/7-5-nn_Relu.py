import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)


class Howjul(nn.Module):
    def __init__(self):
        super(Howjul, self).__init__()
        self.relu1 = ReLU(inplace=False) # 不进行原地替换

    def forward(self, x):
        output = self.relu1(x)
        return output


howjul = Howjul()
output = howjul(input)
print(output)
