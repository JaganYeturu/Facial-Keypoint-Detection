## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# spend more time here
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, 32, 5) # Convolutional Layer 1
        self.pool1 = nn.MaxPool2d(2, 2)  # Convolutional Layer 1
        self.norm1 = nn.BatchNorm2d(32)  # Convolutional Layer 1
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, 5) # Convolutional Layer 2
        self.pool2 = nn.MaxPool2d(2, 2)   # Convolutional Layer 2
        self.norm2 = nn.BatchNorm2d(64)   # Convolutional Layer 2
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(64, 128, 5)  # Convolutional Layer 3
        self.pool3 = nn.MaxPool2d(2, 2)     # Convolutional Layer 3
        self.norm3 = nn.BatchNorm2d(128)    # Convolutional Layer 3
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(128, 256, 3)  # Convolutional Layer 4
        self.pool4 = nn.MaxPool2d(2, 2)    # Convolutional Layer 4
        self.norm4 = nn.BatchNorm2d(256)    # Convolutional Layer 4
        # to do  Convolutional Layer 5
        
        self.ll1 = nn.Linear(11 * 11 * 256, 1000)
        self.ll2 = nn.Linear(1000, 1000)
        self.ll3 = nn.Linear(1000, 136)

	# Comment added
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool1(self.norm1(F.relu(self.conv1(x))))
        x = self.pool2(self.norm2(F.relu(self.conv2(x))))
        x = self.pool3(self.norm3(F.relu(self.conv3(x))))
        x = self.pool4(self.norm4(F.relu(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.ll1(x))
        x = F.relu(self.ll2(x))
        x = self.ll3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
