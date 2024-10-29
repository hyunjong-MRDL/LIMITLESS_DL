import torch
import torch.nn as nn
import timm
from basic_setup import hyperparameters

image_dimension = hyperparameters["TYPE"]

class ASD_Model(nn.Module):
    if image_dimension == 3:  # 3D voxels
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=3, stride=1)
            self.layer2 = nn.Conv3d(in_channels=2, out_channels=2**2, kernel_size=3, stride=1)
            self.layer3 = nn.Conv3d(in_channels=2**2, out_channels=2**3, kernel_size=3, stride=1)
            
            self.activation = nn.ReLU()
            self.maxpool = nn.MaxPool3d(2, 2)
            self.fc = nn.Linear(in_features=8*6*14*14, out_features=2)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.activation(self.maxpool(self.layer1(x)))
            x = self.activation(self.maxpool(self.layer2(x)))
            x = self.activation(self.maxpool(self.layer3(x)))

            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.softmax(x)

            return x
    
    elif image_dimension == 2:  # 2D images
        def __init__(self):
            super().__init__()
            self.model = timm.create_model("resnet50", pretrained=True)
            self.fc = nn.Linear(in_features=1000, out_features=2)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.model(x)
            x = self.fc(x)
            x = self.softmax(x)

            return x

model = ASD_Model()