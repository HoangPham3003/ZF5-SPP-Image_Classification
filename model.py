import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZF5NetNoSPP(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ZF5NetNoSPP, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn_1 = nn.LocalResponseNorm(size=96)
        
        self.conv_2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding='valid')
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn_2 = nn.LocalResponseNorm(size=256)
        
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding='same')

        self.conv_4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding='same')

        self.conv_5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding='same')

        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc_1 = nn.Linear(256*6*6, 4096)

        self.fc_2 = nn.Linear(4096, 4096)

        self.fc_3 = nn.Linear(4096, num_classes)


    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        x = self.lrn_1(x)

        x = self.conv_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        x = self.lrn_2(x)

        x = self.conv_3(x)
        x = F.relu(x)

        x = self.conv_4(x)
        x = F.relu(x)

        x = self.conv_5(x)
        x = F.relu(x)

        x = self.maxpool_3(x)

        x = x.reshape(x.shape[0], -1)

        x = F.dropout(x, p=0.5, inplace=False)
        x = self.fc_1(x)
        x = F.relu(x, inplace=False)
        
        x = F.dropout(x, p=0.5, inplace=False)
        x = self.fc_2(x)
        x = F.relu(x, inplace=False)
        
        x = F.dropout(x, p=0.5, inplace=False)
        x = self.fc_3(x)

        return x
    

class SPPModule(nn.Module):
    def __init__(self, conv5_dim=13, bins=[1]):
        super(SPPModule, self).__init__()
        
        """
            input:
                conv5_dim: shape of output feature map of conv_5 (bs, c, w, h)
                bins: pyramid levels [6, 3, 2, 1]
        """
        
        self.bins = bins
        
        self.n = conv5_dim
        
        self.output_dim = 0
        self.module_list = []
        
        for bin in self.bins:
            self.output_dim += bin * bin * 256
            
            win_size = int(np.ceil(self.n / bin))
            stride = int(np.floor(self.n / bin))
            
            pad = 0
            if self.n == 10 and bin == 6:
                pad = 1
                stride += 1
            
            self.module_list.append(
                nn.MaxPool2d(kernel_size=win_size, stride=stride, padding=pad)
            )
        self.module_list = nn.ModuleList(self.module_list)
        
    def get_output_dim(self):
        return self.output_dim
    
    def forward(self, x):
        outputs = []
        
        for module in self.module_list:
            pool_result = module(x)
            pool_result = pool_result.reshape(pool_result.shape[0], -1)
            outputs.append(pool_result)
        outputs = torch.cat(tensors=outputs, dim=1)
        
        return outputs


class ZF5NetSPP(nn.Module):
    def __init__(self, input_channels=3, crop_dim=224, num_classes=101):
        super(ZF5NetSPP, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn_1 = nn.LocalResponseNorm(size=96)
        
        self.conv_2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding='valid')
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn_2 = nn.LocalResponseNorm(size=256)
        
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding='same')

        self.conv_4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding='same')

        self.conv_5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding='same')
        
        self.spp = None
        if crop_dim == 224:
            self.spp = SPPModule(conv5_dim=13, bins=[1, 2, 3, 6])
        elif crop_dim == 180:
            self.spp = SPPModule(conv5_dim=10, bins=[1, 2, 3, 6])
        self.output_dim_spp = self.spp.get_output_dim()

        self.fc_1 = nn.Linear(self.output_dim_spp, 4096)

        self.fc_2 = nn.Linear(4096, 4096)

        self.fc_3 = nn.Linear(4096, num_classes)


    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        x = self.lrn_1(x)

        x = self.conv_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        x = self.lrn_2(x)

        x = self.conv_3(x)
        x = F.relu(x)

        x = self.conv_4(x)
        x = F.relu(x)

        x = self.conv_5(x)
        x = F.relu(x)

        x = self.spp(x)

        x = F.dropout(x, p=0.5, inplace=False)
        x = self.fc_1(x)
        x = F.relu(x, inplace=False)
        
        x = F.dropout(x, p=0.5, inplace=False)
        x = self.fc_2(x)
        x = F.relu(x, inplace=False)
        
        x = F.dropout(x, p=0.5, inplace=False)
        x = self.fc_3(x)

        return x       