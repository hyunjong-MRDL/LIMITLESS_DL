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
test_ctrl_path = list(set(ctrl_path)-set(train_ctrl_path))
train_ASD_path = list(map(lambda x: ASD_path[x], random.sample(range(0, len(ASD_path)), round(len(ASD_path)*(1-portion)))))
test_ASD_path = list(set(ASD_path)-set(train_ASD_path))

train_path = train_ctrl_path + train_ASD_path
train_label = len(train_ctrl_path)*[[0,1]] + len(train_ASD_path)*[[1,0]]
test_path = test_ctrl_path + test_ASD_path
test_label = len(test_ctrl_path)*[[0,1]] + len(test_ASD_path)*[[1,0]]

basic_trans_3D = torchvision.transforms.Compose([
    tio.transforms.RandomAffine(
        scales=(1,1),
        degrees=0,
    )
])
horz_flip_3D = tio.transforms.Compose([
    tio.transforms.RandomFlip(axes=0, flip_probability=1.0),
])
rotate_90_3D = tio.transforms.Compose([
    tio.transforms.RandomAffine(scales=(1,1), degrees=90),
])
elastic_deformation = tio.transforms.Compose([
    tio.transforms.RandomElasticDeformation(
        num_control_points=7,
        locked_borders=2
    ),
])
gaussian_blur_3D = tio.transforms.Compose([
    tio.transforms.RandomBlur(std=(0,2)),
])
aug_list_3D = [horz_flip_3D, rotate_90_3D, elastic_deformation, gaussian_blur_3D]

# transform_monai = monai.transforms.Compose([
#     monai.transforms.RandAffined(keys=['image', 'label'],
#                                  mode=('bilinear', 'nearest'),
#                                  prob=1.0,
#                                  spatial_size=(300, 300, 50),
#                                  translate_range=(40, 40, 2),
#                                  rotate_range=(np.pi/36, np.pi/36, np.pi/4)),
#     monai.transforms.Resized(keys=['img'], spatial_size=(96, 224, 224))
# ])

basic_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
])
horz_flip = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(p=1),
    torchvision.transforms.ToTensor()
])
rotate_90 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomRotation(90),
    torchvision.transforms.ToTensor()
])
rotate_180 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomRotation(180),
    torchvision.transforms.ToTensor()
])
gaussian_blur = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.GaussianBlur(kernel_size=(7,7), sigma=(0.1, 0.2)),
    torchvision.transforms.ToTensor()
])
aug_list = [horz_flip, rotate_90, rotate_180, gaussian_blur]

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
            zero_mask = (denominator == 0)
            result = np.where(zero_mask, 0.0, numerator / denominator)
            return result
        
        def __LbyR__(self, voxel):
            x, _, _ = voxel.shape
            left = voxel[:int(x/2), ...]
            right = voxel[int(x/2):, ...]
            return self.__safe_divide__(left, right)

        def __getitem__(self, index):
            path = self.path[index]
            label = np.array(self.label[index])

            voxel = self.__crop__(self.__get_voxel__(path))
            voxel = self.__LbyR__(voxel)

            if self.transform is not None:
                voxel = self.transform(torch.from_numpy(voxel).unsqueeze(0))

            return voxel, torch.from_numpy(label)
    
    elif image_dimension == 2:  # 2D images
        def __init__(self, path, label, transform=None):
            self.path = path
            self.label = label
            self.transform = transform
        
        def __len__(self):
            return len(self.path)
        
        def __get_img__(self, path):
            sagittal = nib.load(path).get_fdata()[128, 16:240, 16:240]  # index=1
            axial = nib.load(path).get_fdata()[16:240, 128, 16:240]  # index=2
            coronal = nib.load(path).get_fdata()[16:240, 16:240, 128]  # index=3
            return sagittal, axial, coronal
        
        def __normalize__(self, image):
            minimum = np.min(image)
            maximum = np.max(image)
            return ( (image - minimum) / (maximum - minimum) * 255 ).astype(np.uint8)
        
        def __getitem__(self, index):
            path = self.path[index]
            label = np.array(self.label[index])

            sagittal, axial, coronal = self.__get_img__(path)
            sag_crop = crop_brain(self.__normalize__(sagittal))
            axi_crop = crop_brain(self.__normalize__(axial))
            cor_crop = crop_brain(self.__normalize__(coronal))

            if self.transform is not None:
                sag = self.transform(PIL.Image.fromarray(sag_crop))  # The input image for torchvision.transforms SHOULD be in "PIL.Image" format
                axi = self.transform(PIL.Image.fromarray(axi_crop))
                cor = self.transform(PIL.Image.fromarray(cor_crop))
            
            return torch.concat((sag, axi, cor), dim=0), torch.from_numpy(label)

if image_dimension == 3:
    train_dataset = ASD_Dataset(train_path, train_label, transform=basic_trans_3D)
    for i in range(len(aug_list_3D)):
        train_dataset += ASD_Dataset(train_path, train_label, transform=aug_list_3D[i])
    test_dataset = ASD_Dataset(test_path, test_label, transform=basic_trans_3D)

elif image_dimension == 2:
    train_dataset = ASD_Dataset(train_path, train_label, transform=basic_transform)
    for i in range(len(aug_list)):
        train_dataset += ASD_Dataset(train_path, train_label, transform=aug_list[i])

    test_dataset = ASD_Dataset(test_path, test_label, transform=basic_transform)