import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.modules.loss import MSELoss

class Learn_PPF(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in =size_in
        self.size_out=size_out
        weights = torch.Tensor(size_out,size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        
        nn.init.uniform_(self.weights,a=-(1/size_in)**0.5,b=(1/size_in)**0.5) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        

    def forward(self, points):
        '''
        centers tensor shape [B,S,N](N-N element of the point)
        points tensor shape [B,S,N](M-M points)
        first three elements are set to x y z nx ny nzcoordinates
        '''
        pointsA=points
        
        #avoid same index
        noOfPoints = points.shape[1]
        idxOrigin = torch.arange(noOfPoints)
        idxRand = torch.randperm(noOfPoints)
        sameidx = torch.where(idxOrigin==idxRand)
        idxRand[sameidx] = idxRand[sameidx]-1

        pointsB=points[:,idxRand,:]
        d0 = pointsA[:,:,0] - pointsB[:,:,0]
        d1 = pointsA[:,:,1] - pointsB[:,:,1]
        d2 = pointsA[:,:,2] - pointsB[:,:,2]
        ppf0 = torch.sqrt(torch.square(d0) + torch.square(d1) + torch.square(d2))
        print(torch.where(ppf0==0))
        ppf1 = (pointsA[:,:,3]*d0 + pointsA[:,:,4]*d1 + pointsA[:,:,5]*d2)/(ppf0 * torch.sqrt(torch.square(pointsA[:,:,3]) + torch.square(pointsA[:,:,4]) + torch.square(pointsA[:,:,5])))
        ppf2 = (pointsB[:,:,3]*d0 + pointsB[:,:,4]*d1 + pointsB[:,:,5]*d2)/(ppf0 * torch.sqrt(torch.square(pointsB[:,:,3]) + torch.square(pointsB[:,:,4]) + torch.square(pointsB[:,:,5])))
        ppf3 = (pointsA[:,:,3]*pointsB[:,:,3] + pointsA[:,:,4]*pointsB[:,:,4] + pointsA[:,:,5]*pointsB[:,:,5])/(torch.sqrt(torch.square(pointsA[:,:,3]) + torch.square(pointsA[:,:,4]) + torch.square(pointsA[:,:,5])) * torch.sqrt(torch.square(pointsB[:,:,3]) + torch.square(pointsB[:,:,4]) + torch.square(pointsB[:,:,5])))
        input = torch.Tensor(pointsA.shape[0],pointsA.shape[1],4)
        output = torch.Tensor(pointsA.shape[0],pointsA.shape[1],self.size_out)
        if self.weights.is_cuda:
            cudaDevice = self.weights.device
            input = input.to(cudaDevice)
            output = output.to(cudaDevice)
        input[:,:,0] = ppf0
        input[:,:,1] = ppf1
        input[:,:,2] = ppf2
        input[:,:,3] = ppf3
        #print(torch.where(torch.isnan(input)==True))
        for i in range(pointsA.shape[0]):
            #print(input[i].shape)
            #print(self.weights.t().shape)
            output[i] = torch.mm(input[i], self.weights.t())
            output[i] = torch.add(output[i], self.bias) 
        #print(torch.where(torch.isnan(self.weights)==True))
        #print(torch.where(torch.isnan(output)==True))
        output[torch.where(torch.isnan(output)==True)]=0
        return output# w times x + b

class Learn_PFH(nn.Module):
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
        w_times_x= torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b

#Test Code

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv = nn.Conv2d(1, 128, 3)
        # self.linear = nn.Linear(256, 2)
        self.linear = Learn_PPF(4,128)

    def forward(self, pointsA):
        #x = self. conv(x)
        #x = x.view(-1, 256)
        return self.linear(pointsA)
        
if __name__ == '__main__':
    print('debug0')
    torch.manual_seed(0)  #  for repeatable results
    #initilize model, loss and optimizer
    basic_model = BasicModel()
    optimizer = torch.optim.SGD(basic_model.parameters(), lr=1)
    MSEloss=torch.nn.L1Loss()
    batch=10

    #first forward and bp
    pointsA = torch.rand(batch,512,6)
    pointsB = torch.rand(batch,512,6)
    target = torch.rand(batch,512,3)
    output = basic_model(pointsA)
    optimizer.zero_grad()
    loss=MSEloss(target, output)
    loss.backward()
    optimizer.step()
    print(loss)

    #second forward and bp
    pointsA = torch.rand(batch,512,6)
    pointsB = torch.rand(batch,512,6)
    target = torch.rand(batch,512,3)
    output = basic_model(pointsA)
    optimizer.zero_grad()
    loss = MSEloss(target, output)
    loss.backward()
    optimizer.step()
    print(loss)
    print('debug1')
    print('Forward computation thru model:', basic_model(pointsA).shape)