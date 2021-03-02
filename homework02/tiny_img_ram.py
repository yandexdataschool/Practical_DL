import os
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision import transforms

import cv2
from PIL import Image

class TinyImagenetRAM(Dataset):
    def __init__(self, root, transform=transforms.ToTensor()):
        super(TinyImagenetRAM, self).__init__()
        
        self.root = root
        self.classes = sorted([item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))])
        self.class_to_idx = {item: index for index, item in enumerate(self.classes)}
        
        self.transform = transform
        self.images, self.targets = [], []
        for index, item in tqdm(enumerate(self.classes), total=len(self.classes), desc=self.root):
            path = os.path.join(root, item, 'images')
            for name in os.listdir(path):
                image = cv2.cvtColor(cv2.imread(os.path.join(path, name)), cv2.COLOR_BGR2RGB)
                self.images.append(Image.fromarray(image))
                self.targets.append(index)
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.transform(self.images[index])
        target = self.targets[index]
        return image, target