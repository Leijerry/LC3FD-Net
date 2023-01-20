import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils_ppf import PointNetSetAbstractionMsg, PointNetSetAbstraction
import time
import os
from models.threedlfd import Learn_PPF
import torch

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        ppf_channel=128
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        #self.ppf = Learn_PPF(4,ppf_channel)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            xyz_norm=xyz
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        #ppf=self.ppf(xyz_norm.permute(0,2,1)).permute(0,2,1)
        #print(torch.where(torch.isnan(ppf)==True))
        l1_xyz, l1_points, l1_xyz_norm= self.sa1(xyz, norm, xyz_norm)
        #l1_ppf=self.ppf(l1_xyz_norm).permute(0,2,1)
        #print(torch.where(torch.isnan(l1_ppf)==True))
        l1_xyz_norm=l1_xyz_norm.permute(0,2,1)
        l2_xyz, l2_points, l2_xyz_norm = self.sa2(l1_xyz, l1_points, l1_xyz_norm)
        #l2_ppf=self.ppf(l2_xyz_norm).permute(0,2,1)
        l2_xyz_norm = l2_xyz_norm.permute(0,2,1)
        #l2_xyz=self.ppf(l2_xyz)
        l3_xyz, l3_points, l3_xyz_norm = self.sa3(l2_xyz, l2_points, l2_xyz_norm)
        #print(l1_xyz)
        #print(l2_xyz)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x

if __name__ == '__main__':
    import  torch
    model = get_model(10)
    #input shape sequence(batchSize, (x,y,z), numberOfPoints)
    #initial pointnet input size: 1024
    #our input size: 32768
    batch_size=2
    print("--------Pointnet++ Speed Test---------")
    print('Input Points: 1024')
    xyz = torch.rand(batch_size, 6, 1024)
    xyz=xyz.to('cuda')
    model.to('cuda')
    tik=time.perf_counter()
    with torch.no_grad():
        output = model(xyz)
    tok=time.perf_counter()
    print(output.shape)
    print("Pointnet++ time: ", (tok-tik),"s")
    
    print("-----------------")
    print('Input Points: 32768')
    xyzl = torch.rand(batch_size, 6, 32768)
    xyzl=xyzl.to('cuda')
    model.to('cuda')
    tik=time.perf_counter()
    with torch.no_grad():
        output1 = model(xyzl)
    tok=time.perf_counter()
    print("Pointnet++ time: ", (tok-tik),"s")
    
