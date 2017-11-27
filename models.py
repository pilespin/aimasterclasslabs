import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # self.fc0 = nn.Linear(28*28, 28*28)
        # self.fc1 = nn.Linear(28*28, 20*20)
        # self.fc2 = nn.Linear(20*20, 20*20)
        # self.fc3 = nn.Linear(20*20, 27)

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 4)
        self.conv2 = nn.Conv2d(6, 16, 4)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 27)

        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)


    def forward(self, x):

    	# # Max pooling over a (2, 2) window
     #    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
     #    # If the size is a square you can only specify a single number
     #    x = F.max_pool2d(F.relu(self.conv2(x)), 2)

     #    # x = x.view(-1, self.num_flat_features(x))
     #    x = self.fc1(x.view(x.size(0), -1))
        
     #    x = F.relu(self.fc1(x))
     #    x = F.relu(self.fc2(x))
     #    x = self.fc3(x)
     #    return x

    	# Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

    	# import ipdb;ipdb.set_trace()
        # x = self.fc0(x.view(x.size(0), -1))
        # x = self.fc1(x.view(x.size(0), -1))
        # x = self.fc2(x.view(x.size(0), -1))
        # x = self.fc3(x.view(x.size(0), -1))

        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.fc1(x.view(x.size(0), -1))
        # x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))


        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        
        return F.log_softmax(x)

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features
