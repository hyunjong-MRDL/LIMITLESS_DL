import os, PIL
import pandas as pd
from torch.utils.data import Dataset

"""Data Loading"""
def data_path(root, train):
    if train:
        im_path = f"{root}Train_images/"
    else:
        im_path = f"{root}Test_images/"
    return os.listdir(im_path)

def label_path(root, train):
    if train:
        label = f"{root}Train_label.csv"
    else:
        label = f"{root}Test_label.csv"
    return label

def read_label(file):
    label_list = []
    dictionary = pd.read_csv(file)
    labels = dictionary.drop(["ID", "img_path"], axis=1)
    for i in range(len(labels)):
        label_list.append(list(labels.iloc[i, :]))
    return label_list

"""Data Preprocessing"""
class PuzzleDataset(Dataset):
    def __init__(self, path, label, transform):
        super().__init__()
        self.path = path
        self.label = label
        self.transform = transform
    
    def __len__(self):
        return len(self.path)
    
    def __get_img__(self, path):
        image = PIL.Image.open(path)
        return image
    
    def __get_item__(self, index):
        image = self.__get_img__(self.path[index])
        label = self.label[index]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label