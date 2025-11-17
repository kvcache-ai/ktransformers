import torch
import torch.nn as nn
from torchviz import make_dot

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()

input_tensor = torch.randn(1, 10)

output = model(input_tensor)

dot = make_dot(output, params=dict(model.named_parameters()))
dot.render('simple_net', format='svg', cleanup=True)    