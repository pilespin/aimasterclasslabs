import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 4)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.conv3 = nn.Conv2d(16, 32, 2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 27)


    def forward(self, x):

        x = F.max_pool2d(F.tanh(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.tanh(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.tanh(self.conv3(x)), (2, 2))
        
        x = self.fc1(x.view(x.size(0), -1))
        # x = self.fc1(x)
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        
        return F.log_softmax(x)
