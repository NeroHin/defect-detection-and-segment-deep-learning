import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import json
import numpy as np
from PIL import Image

sys.path.append(os.path.realpath('..'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_set_dir = '../defect-detection-and-segment-deep-learning/class_data/Train'
test_set_dir = '../defect-detection-and-segment-deep-learning/class_data/Val'

# Define the transformations to apply to the images
transform = torchvision.transforms.Compose([
    # resize the image to 224x224
    torchvision.transforms.Resize((224, 224)),
    
    # convert the rgb image to grayscale
    torchvision.transforms.Grayscale(num_output_channels=1),
    # convert the image to a tensor
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# Define the custom dataset class
class DefectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, mask:bool=False):
        self.root_dir = root_dir
        self.transform = transform
        self.mask = mask
        self.class_names = ['powder_uncover', 'powder_uneven', 'scratch']
        self.types = ['image']
        self.image_filenames = []
        # bounding box
        self.labels = []
        self.mask_filenames = []
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            # concatenate the image, label and mask directories
            for type_name in self.types:
                type_dir = os.path.join(class_dir, type_name)
                for filename in os.listdir(type_dir):
                    if filename.endswith('.png'):
                        self.image_filenames.append(os.path.join(type_dir, filename))
                        # read the label file for bounding box position

                        self.mask_filenames.append(os.path.join(type_dir.replace('image', 'mask'), filename.replace('.png', '.png')))
                        self.labels.append(class_name)
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx])
        mask = Image.open(self.mask_filenames[idx]).convert('L')
        
    
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.mask == True:    
            return image, mask, label
        else:
            return image, label
    
# Load the training and test datasets
trainset = DefectDetectionDataset(root_dir=train_set_dir, transform=transform)
testset = DefectDetectionDataset(root_dir=test_set_dir, transform=transform)

batch_size = 32

# Define the data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

model = torchvision.models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(trainset.class_names))

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


