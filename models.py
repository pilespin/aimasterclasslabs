import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 2)
        self.conv2 = nn.Conv2d(6, 12, 2)
        self.conv3 = nn.Conv2d(12, 18, 2)
        self.conv4 = nn.MaxPool2d(2)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(450, 300)
        self.fc2 = nn.Linear(2592, 500)
        self.fc3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(500, 27)


    def forward(self, x):

        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        # x = F.max_pool2d(F.tanh(self.conv3(x)), (2, 2))
        
        x = self.fc2(x.view(x.size(0), -1))
        # x = self.fc1(x)
        # x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        
        return F.log_softmax(x)
