import torch
import torch.nn as nn
import math
import numpy as np
import os

class Learn_PPF(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_in)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x= torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b

class Learn_FPFH(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def forward(self, x):
        #print(x.shape)
        #print(self.weights.shape)
        w_times_x= torch.mm(x, self.weights.t())
        print(self.weights[0][0:10])
        return torch.add(w_times_x, self.bias)    # w times x + b

#Test Code

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 128, 3)
        # self.linear = nn.Linear(256, 2)
        self.linear = Learn_FPFH(256, 2)

    def forward(self, x):
        x = self. conv(x)
        #print(self. conv.weight)
        #print(x.shape)
        x = x.view(-1, 256)
        return self.linear(x)
        
if __name__ == '__main__':
    print('debug0')
    torch.manual_seed(0)  #  for repeatable results
    basic_model = BasicModel()
    basic_model.train()
    x = torch.rand((1,1,3,4))
    target=torch.ones((1,2))
    l1loss=torch.nn.L1Loss()
    optimizer = torch.optim.SGD(basic_model.parameters(), lr=10)
    #print(x.shape)

    out = basic_model(x)
    #print(out.shape)
    loss=l1loss(out,target)
    print(loss)
    torch.autograd.set_detect_anomaly(True)
    #os.system('pause')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #os.system('pause')
    #inp = np.random.rand(1,1,3,4)
    x = torch.rand((1,1,3,4))
    target = torch.ones((1,2))
    out = basic_model(x)
    optimizer.zero_grad()
    loss=l1loss(out,target)
    print(loss)
    loss.backward()
    optimizer.step()
    print('debug1')
    print('Forward computation thru model:', basic_model(x).shape)