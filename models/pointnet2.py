'''
Based On:
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
Modified by: Yangzheng Wu
'''
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
import time

batch_size=4

class Pointnet2(nn.Module):
    def __init__(self, num_radius):
        super(Pointnet2, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_radius, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        #tik=time.perf_counter()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        #tok=time.perf_counter()
        #print("encoding time:", (tok-tik),"s")
        #print(l1_xyz.size(),l1_points.size(),l2_xyz.size(),l2_points.size(),l3_xyz.size(),l3_points.size(),l4_xyz.size(),l4_points.size())
        
        #tik=time.perf_counter()
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        #tok=time.perf_counter()
        #print("decoding time: ", (tok-tik), "s")

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = Pointnet2(9)
    #input shape sequence(batchSize, (x,y,z), numberOfPoints)
    #initial pointnet input size: 1024
    #our input size: 32768
    
    print("--------Pointnet++ Speed Test---------")
    print('Input Points: 1024')
    xyz = torch.rand(batch_size, 3, 1024)
    xyz=xyz.to('cuda')
    model.to('cuda')
    tik=time.perf_counter()
    with torch.no_grad():
        output = model(xyz)
    tok=time.perf_counter()
    print("Pointnet++ time: ", (tok-tik),"s")
    
    print("-----------------")
    print('Input Points: 32768')
    xyzl = torch.rand(batch_size, 3, 32768)
    xyzl=xyzl.to('cuda')
    model.to('cuda')
    tik=time.perf_counter()
    with torch.no_grad():
        output1 = model(xyzl)
    tok=time.perf_counter()
    print("Pointnet++ time: ", (tok-tik),"s")
    

    #print(output1)
    #print(output2.shape)