import os, random

from glob import glob

from PIL import Image

from torch.utils.data import Dataset

def read_data(dataset_path):
    data = {}
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        if os.path.isdir(file_path):
            data[file] = []
            for img_path in glob(file_path + "/*.jpg"):
                data[file].append(img_path)
    return data

def split_dataset(data, ratio=0.8):
    train_data = {}
    test_data = {}
    classes = []
    for class_name in data.keys():
        classes.append(class_name)
        
        train_data[class_name] = []
        test_data[class_name] = []

        split_index = int(len(data[class_name]) * ratio)
        
        random.shuffle(data[class_name])
        
        for img_path in data[class_name][:split_index]:
                train_data[class_name].append(img_path)
        
        for img_path in data[class_name][split_index:]:
                test_data[class_name].append(img_path)
    
    return train_data, test_data, classes

def get_dataset(data, classes):
    dataset = []
    target = 0
    for class_name in classes:
        for img_path in data[class_name]:
            dataset.append((img_path, target))
        target += 1
    return dataset


class RiceDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, label = self.dataset[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        
        return img, label
