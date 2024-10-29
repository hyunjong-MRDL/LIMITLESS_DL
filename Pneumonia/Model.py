import torch.nn as nn
import timm

class PNEU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Conv2d(1,3,1,1)
        self.model = timm.create_model("resnet50")
        self.fc = nn.Linear(in_features=1000, out_features=3)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.pre_layer(x)
        x = self.model(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x