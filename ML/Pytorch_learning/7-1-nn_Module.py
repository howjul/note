import torch
from torch import nn


class Howjul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

howjul = Howjul()
x = torch.tensor(1.0)
output = howjul(x)
print(output)