import PIL, torch, torchvision
import skimage.transform as sk_transform
import torchio as tio
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

import random
from Readpath import ctrl_path, ASD_path
from basic_setup import seed_everything, hyperparameters
from Preprocessing import get_boundaries, crop_brain

seed_everything(hyperparameters["SEED"])
portion = hyperparameters["TEST_PORTION"]
image_dimension = hyperparameters["TYPE"]

train_ctrl_path = list(map(lambda x: ctrl_path[x], random.sample(range(0, len(ctrl_path)), round(len(ctrl_path)*(1-portion)))))
train_ctrl_label = len(train_ctrl_path)*[[0,1]]
test_ctrl_path = list(set(ctrl_path)-set(train_ctrl_path))

train_ASD_path = list(map(lambda x: ASD_path[x], random.sample(range(0, len(ASD_path)), round(len(ASD_path)*(1-portion)))))
train_ASD_label = len(train_ASD_path)*[[1,0]]
test_ASD_path = list(set(ASD_path)-set(train_ASD_path))

test_path = test_ctrl_path + test_ASD_path
test_label = len(test_ctrl_path)*[[0,1]] + len(test_ASD_path)*[[1,0]]

if image_dimension == 3:
    basic_trans = torchvision.transforms.Compose([
        tio.transforms.RandomAffine(
            scales=(1,1),
            degrees=0,
        )
    ])
    horz_flip = tio.transforms.Compose([
        tio.transforms.RandomFlip(axes=0, flip_probability=1.0),
    ])
    rotate_90 = tio.transforms.Compose([
        tio.transforms.RandomAffine(scales=(1,1), degrees=90),
    ])
    elast_deform = tio.transforms.Compose([
        tio.transforms.RandomElasticDeformation(
            num_control_points=7,
            locked_borders=2
        ),
    ])
    gaussian_blur = tio.transforms.Compose([
        tio.transforms.RandomBlur(std=(0,2)),
    ])

elif image_dimension == 2:
    basic_trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
    ])
    horz_flip = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(p=1),
        torchvision.transforms.ToTensor()
    ])
    rotate_90 = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(90),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    elast_deform = torchvision.transforms.Compose([
        torchvision.transforms.ElasticTransform(alpha=50, sigma=5.0),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    gaussian_blur = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.GaussianBlur(kernel_size=(7,7), sigma=(0.1, 0.2)),
        torchvision.transforms.ToTensor()
    ])

aug_list = [horz_flip, rotate_90, elast_deform, gaussian_blur]

class ASD_Dataset(Dataset):
    if image_dimension == 3:  # 3D voxels
        def __init__(self, path, label, transform=None):
            self.path = path
            self.label = label
            self.transform = transform
        
        def __len__(self):
            return len(self.path)
        
        def __get_voxel__(self, path):  # (256, 256, 256)
            voxel = nib.load(path).get_fdata()  # nib.load()함수는 기본적으로 array proxy(대리인, 간접참조)를 반환
            return voxel                        # 반환된 array proxy에 get_fdata() 메소드를 적용하면 image array를 얻을 수 있다
        
        def __normalize__(self, voxel):
            minimum = np.min(voxel)
            maximum = np.max(voxel)
            return ( (voxel - minimum) / (maximum - minimum) * 255 ).astype(np.uint8)
        
        def __crop__(self, voxel):
            sagittal = voxel[128, ...]
            axial = voxel[:, 128, :]
            coronal = voxel[..., 128]

            sz, sy, w1, h1  = get_boundaries(self.__normalize__(sagittal))
            ax, az, w2, h2 = get_boundaries(self.__normalize__(axial))
            cx, cy, w3, h3 = get_boundaries(self.__normalize__(coronal))
            
            cropped_voxel = sk_transform.resize(voxel[max(ax, cx): max(ax+w2, cx+w3), max(sy, cy):max(sy+h1, cy+h3), max(sz, az):max(sz+w1, az+h2)], (128, 128, 128))
            return cropped_voxel

        def __safe_divide__(self, numerator, denominator):
            denominator = np.where(denominator==0, 1, denominator)
            return ( numerator / denominator )
        
        def __LbyR__(self, voxel):
            _, x, _ = voxel.shape
            left = voxel[:x//2, ...]
            right = np.flip(voxel[x//2:, ...], axis=0)
            return self.__safe_divide__(left, right)

        def __getitem__(self, index):
            path = self.path[index]
            label = np.array(self.label[index])

            voxel = self.__crop__(self.__get_voxel__(path))
            voxel = self.__LbyR__(voxel)

            if self.transform is not None:
                voxel = self.transform(torch.from_numpy(voxel).unsqueeze(0))

            return voxel, torch.from_numpy(np.array(label))
    
    elif image_dimension == 2:  # 2D images
        def __init__(self, path, label, transform=None):
            self.path = path
            self.label = label
            self.transform = transform
        
        def __len__(self):
            return len(self.path)
        
        def __get_voxel__(self, path):  # (256, 256, 256)
            voxel = nib.load(path).get_fdata()
            return voxel
        
        def __normalize__(self, voxel):
            minimum = np.min(voxel)
            maximum = np.max(voxel)
            return ( (voxel - minimum) / (maximum - minimum) * 255 ).astype(np.uint8)
        
        def __crop__(self, voxel):
            sagittal = voxel[128, ...]
            axial = voxel[:, 128, :]
            coronal = voxel[..., 128]

            sag_cropped  = crop_brain(self.__normalize__(sagittal))
            axi_cropped = crop_brain(self.__normalize__(axial))
            cor_cropped = crop_brain(self.__normalize__(coronal))
            return sag_cropped, axi_cropped, cor_cropped
        
        def __getitem__(self, index):
            path = self.path[index]
            label = self.label[index]
            sag, axi, cor = self.__crop__(self.__get_voxel__(path))

            if self.transform is not None:
                sag = self.transform(PIL.Image.fromarray(sag))
                axi = self.transform(PIL.Image.fromarray(axi))
                cor = self.transform(PIL.Image.fromarray(cor))
            
            return torch.cat((sag, axi, cor), dim=0), torch.from_numpy(np.array(label))

train_ctrl_dataset = ASD_Dataset(train_ctrl_path, train_ctrl_label, transform=basic_trans)
train_ASD_dataset = ASD_Dataset(train_ASD_path, train_ASD_label, transform=basic_trans)
for i in range(len(aug_list)):
    train_ctrl_dataset += ASD_Dataset(train_ctrl_path, train_ctrl_label, transform=aug_list[i])
train_dataset = train_ctrl_dataset + train_ASD_dataset
test_dataset = ASD_Dataset(test_path, test_label, transform=basic_trans)