import cv2, PIL, torch, torchvision
from torch.utils.data import Dataset

from Basic_settings import hyperparameters
from Read_path import read_path

im_size = hyperparameters["IMSIZE"]

base_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((im_size, im_size)),
    torchvision.transforms.ToTensor()
])

class PNEU_Dataset(Dataset):
    def __init__(self, path, label, transform=None):
        self.path = path
        self.label = label
        self.transform = transform
    
    def __len__(self):
        return len(self.path)
    
    def __getimg__(self, path):
        image = cv2.imread(path)
        if image.shape[2] == 3:
            image = image[..., 0]
        
        return PIL.Image.fromarray(image)
    
    def __getitem__(self, index):
        path = self.path[index]
        label = self.label[index]

        image = self.__getimg__(path)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.from_numpy(label)

root = "C:/Users/PC00/Desktop/HJ/AICOSS_2023(Pneumonia)/Cropped_Images/"

train_path, train_label = read_path(root, type=0)
test_path, test_label = read_path(root, type=1)
val_path, val_label = read_path(root, type=2)

train_dataset = PNEU_Dataset(train_path, train_label, transform=base_transform)
test_dataset = PNEU_Dataset(test_path, test_label, transform=base_transform)
val_dataset = PNEU_Dataset(val_path, val_label, transform=base_transform)