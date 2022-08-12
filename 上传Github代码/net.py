import torch.nn as nn
import torch
from resnet1D import *





class Net(nn.Module):
    def __init__(self, n_feature, hidden_size, label_size, t):
        super(Net, self).__init__()
        self.type = t
        self.hidden_size = hidden_size
        if self.type == "Linear":
            self.linear_s1_1 = nn.Linear(n_feature - 3, hidden_size)
            self.ReLU_s1_1 = nn.ReLU(True)
            self.linear_s1_2 = nn.Linear(hidden_size, hidden_size)
            self.ReLU_s1_2 = nn.ReLU(True)
            self.linear_s1_3 = nn.Linear(hidden_size, label_size)

            self.linear_s2_1 = nn.Linear(3, hidden_size)
            self.ReLU_s2_1 = nn.ReLU(True)
            self.linear_s2_2 = nn.Linear(hidden_size, hidden_size)
            self.ReLU_s2_2 = nn.ReLU(True)
            self.linear_s2_3 = nn.Linear(hidden_size, label_size)

            self.linear_cat = nn.Linear(2 * label_size, label_size)
        if self.type == "ResNet":
            self.linear_s1 = nn.Linear(n_feature - 4, hidden_size)
            self.linear_s2 = nn.Linear(4, hidden_size)
            self.Conv1d_s1 = nn.Conv1d(1, 64, 1, 1)
            self.Conv1d_s2 = nn.Conv1d(1, 64, 1, 1)
            self.BN_s1 = nn.BatchNorm1d(64)
            self.BN_s2 = nn.BatchNorm1d(64)
            self.relu_s1 = nn.PReLU(num_parameters=64)
            self.relu_s2 = nn.PReLU(num_parameters=64)
            self.ResNet_s1 = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type='prelu')
            self.ResNet_s2 = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type='prelu')
            self.linear_cat = nn.Linear(2 * 512, label_size)



    def forward(self, x):
        s1 = x[:, :-4]
        s2 = x[:, -4:]
        s = None
        if self.type == "Linear":
            s1 = self.linear_s1_1(s1)
            s1 = self.ReLU_s1_1(s1)
            s1 = self.linear_s1_2(s1)
            s1 = self.ReLU_s1_2(s1)
            s1 = self.linear_s1_3(s1)

            s2 = self.linear_s2_1(s2)
            s2 = self.ReLU_s2_1(s2)
            s2 = self.linear_s2_2(s2)
            s2 = self.ReLU_s2_2(s2)
            s2 = self.linear_s2_3(s2)



        if self.type == "ResNet":
            s1 = self.linear_s1(s1)
            s1 = s1.reshape(-1, 1, self.hidden_size)
            s1 = self.Conv1d_s1(s1)
            s1 = self.BN_s1(s1)
            s1 = self.relu_s1(s1)
            s1 = self.ResNet_s1(s1)
            s1 = s1.reshape(-1, 512)

            s2 = self.linear_s2(s2)
            s2 = s2.reshape(-1, 1, self.hidden_size)
            s2 = self.Conv1d_s2(s2)
            s2 = self.BN_s2(s2)
            s2 = self.relu_s2(s2)
            s2 = self.ResNet_s2(s2)
            s2 = s2.reshape(-1, 512)
            # print(s1.shape) # torch.Size([10, 512, 1])


        s = torch.cat((s1, s2), dim=-1)
        s = self.linear_cat(s)

        return s