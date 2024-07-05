import cv2
import torch
import numpy as np
import advertisments as ads
import torch.nn as nn

model_parameters = {}
model_parameters['densenet121'] = [6, 12, 24, 16]

# growth rate
k = 32
compression_factor = 0.5

class DenseLayer(nn.Module):
    def __init__(self, in_channels):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=4*k, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=4*k)
        self.conv2 = nn.Conv2d(in_channels=4*k, out_channels=k, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        in_x = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.cat([in_x, x], 1)
        return x

class DenseBlock(nn.Module):
    def __init__(self, layers_count, in_channel):
        super(DenseBlock, self).__init__()
        self.layers_count = layers_count
        self.layers = nn.ModuleList([DenseLayer(in_channels=in_channel + k * num) for num in range(layers_count)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression_factor):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels * compression_factor), kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        return x

class Densenet(nn.Module):
    def __init__(self, in_channel, classes):
        super(Densenet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layers = nn.ModuleList()
        dense_block_inchannel = 64
        dense_121_layers = model_parameters['densenet121']

        for num in range(len(dense_121_layers) - 1):
            self.layers.add_module(f"DenseBlock_{num + 1}", DenseBlock(dense_121_layers[num], dense_block_inchannel))
            dense_block_inchannel = int(dense_block_inchannel + k * dense_121_layers[num])
            self.layers.add_module(f"TransitionLayer_{num + 1}", TransitionLayer(dense_block_inchannel, compression_factor))
            dense_block_inchannel = int(dense_block_inchannel * compression_factor)

        # Final Layer
        self.layers.add_module(f"DenseBlock_{num + 2}", DenseBlock(dense_121_layers[-1], dense_block_inchannel))
        dense_block_inchannel = int(dense_block_inchannel + k * dense_121_layers[-1])

        self.bn2 = nn.BatchNorm2d(num_features=dense_block_inchannel)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=dense_block_inchannel, out_features=classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)
        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Reference : https://medium.com/@karuneshu21/implement-densenet-in-pytorch-46374ef91900



## VGG13 

import torch.nn as nn
import torch.nn.functional as F

class VGG13(nn.Module):
    def __init__(self, num_classes):
        super(VGG13, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
## Alexnet Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class AlaxNet_custom(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(AlaxNet_custom, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=out_classes)

    def forward(self, x):
        img = self.bn1(self.conv1(x))
        img = self.relu(img)
        img = self.maxPool(img)
        # 2nd
        img = self.bn2(self.conv2(img))
        img = self.relu(img)
        img = self.maxPool(img)
        # 3rd
        img = self.bn3(self.conv3(img))
        img = self.relu(img)
        # 4th
        img = self.bn4(self.conv4(img))
        img = self.relu(img)
        # 5th
        img = self.bn5(self.conv5(img))
        img = self.relu(img)
        img = self.maxPool(img)
        img = torch.flatten(img, 1)
        # 6th
        img = self.dropout(img)
        img = self.fc1(img)
        img = self.relu(img)
        # 7th
        img = self.dropout(img)
        img = self.fc2(img)
        img = self.relu(img)
        # output
        img = self.fc3(img)
        return img

# Reference: https://blog.paperspace.com/alexnet-pytorch/
# Reference: https://github.com/dansuh17/alexnet-pytorch/blob/d0c1b1c52296ffcbecfbf5b17e1d1685b4ca6744/model.py#L40

# Resnet 18

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):

        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.dropout = nn.Dropout(0.38)

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        x = self.dropout(x)
        return x



class ResNet(nn.Module):
  
  def __init__(self, block, layers, image_channels, num_classes):
    super(ResNet, self).__init__()
    self.expansion = 1 # expansion factor is 1 for resnet 18, it is 4 for resnet 50,101 and 152
    self.in_channels = 64
    self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) # Extracts the low level features from input image
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # Feature map size of 112 * 112
    self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1) # Two convolutions with 64 outputs
    self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2) # Two convolution with 128 outputs
    self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2) # Two convolutions with 256 outputs
    self.layer4 = self.make_layer(block, layers[3], out_channels=512, stride=2) # Two convolution with 512 outputs
    # Feature map size of 7 * 7
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * self.expansion, num_classes)
# Increasing the filter size will help us to create more deep model and capture complex models
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x,1)
    x = self.fc(x)

    return x

  def make_layer(self, block, num_residual_blocks, out_channels, stride):
    identity_downsample = None # initially identity_downsample is none
    layers = []

    if stride != 1 or self.in_channels != self.expansion * out_channels:
        identity_downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels)
        )
        # when above case occurs we create a identity_downsample with 1*1 conv layer to match number of channels

    layers.append(block(self.in_channels, out_channels, identity_downsample, stride)) # append the first residual block with matched dimensions to layers
    self.in_channels = out_channels * self.expansion # update input channels

    for i in range(num_residual_blocks - 1): # append remaining residual blocks
        layers.append(block(self.in_channels, out_channels))

    return nn.Sequential(*layers)

def ResNet18(img_channel=3, num_classes=10):
    return ResNet(Block, [2, 2, 2, 2], img_channel, num_classes)