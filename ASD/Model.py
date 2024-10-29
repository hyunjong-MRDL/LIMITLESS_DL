import torch
import torch.nn as nn
import timm
from basic_setup import hyperparameters

image_dimension = hyperparameters["TYPE"]

class ASD_Model(nn.Module):
    if image_dimension == 3:  # 3D voxels
        def __init__(self):
            super().__init__()
            self.InitialBlock = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=2, kernel_size=1, stride=1),  # Channel -> DOUBLE (1x1 Conv)
                nn.BatchNorm3d(num_features=2),
                nn.ReLU(),
                nn.MaxPool3d(2, 2)  # Img_size -> HALF
            )

            self.ResBlock1 = nn.Sequential(
                nn.Conv3d(2, 2, 1, 1),  # K=1, S=1, P=0 (O = I) -> just activation
                nn.BatchNorm3d(2),
                nn.ReLU(),
                nn.Conv3d(2, 4, 7, 2, 3),  # K=7, S=2, P=3 (O = I/2)
                nn.BatchNorm3d(4),
                nn.ReLU(),
                nn.Conv3d(4, 4, 1, 1),  # K=1, S=1, P=0 (O = I) -> just activation
                nn.BatchNorm3d(4),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )

            self.ResBlock2 = nn.Sequential(
                nn.Conv3d(4, 4, 1, 1),  # K=1, S=1, P=0 (O = I) -> just activation
                nn.BatchNorm3d(4),
                nn.ReLU(),
                nn.Conv3d(4, 8, 7, 2, 3),  # K=7, S=2, P=3 (O = I/2)
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.Conv3d(8, 8, 1, 1),  # K=1, S=1, P=0 (O = I) -> just activation
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )

            self.ResBlock3 = nn.Sequential(
                nn.Conv3d(8, 8, 1, 1),  # K=1, S=1, P=0 (O = I) -> just activation
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.Conv3d(8, 16, 7, 2, 3),  # K=7, S=2, P=3 (O = I/2)
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.Conv3d(16, 16, 1, 1),  # K=1, S=1, P=0 (O = I) -> just activation
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )

            self.SkipConnection1 = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=4, kernel_size=1, stride=1),
                nn.BatchNorm3d(4),
                nn.ReLU(),
                nn.MaxPool3d(2, 2),
                nn.Dropout(p=0.5)
            )

            self.SkipConnection2 = nn.Sequential(
                nn.Conv3d(in_channels=4, out_channels=8, kernel_size=1, stride=1),
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.MaxPool3d(2, 2),
                nn.Dropout(p=0.5)
            )

            self.SkipConnection3 = nn.Sequential(
                nn.Conv3d(in_channels=8, out_channels=16, kernel_size=1, stride=1),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d(2, 2),
                nn.Dropout(p=0.5)
            )
            
            self.activation = nn.ReLU()
            self.maxpool = nn.MaxPool3d(2, 2)
            self.fc = nn.Linear(in_features=16*4*4*2, out_features=2)
            self.softmax = nn.Softmax()
        
        def forward(self, x):
            x = self.InitialBlock(x)
            x = self.ResBlock1(x) + self.SkipConnection1(x)
            x = self.ResBlock2(x) + self.SkipConnection2(x)
            x = self.ResBlock3(x) + self.SkipConnection3(x)

            x = self.maxpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.softmax(x)

            return x
    
    elif image_dimension == 2:  # 2D images
        # from huggingface_hub import cached_assets_path
        # assets_path = cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
        def __init__(self):
            super().__init__()
            self.model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
            self.fc = nn.Linear(in_features=1000, out_features=2)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.model(x)
            x = self.fc(x)
            x = self.softmax(x)

            return x

model = ASD_Model()