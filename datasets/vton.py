from io import BytesIO
import os
import lmdb
from PIL import Image
from torch.utils.data import Dataset

class VTON(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.path = path
        self.items = os.listdir(path)
        if not self.path:
            raise IOError('Cannot open lmdb dataset', path)



        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.path, self.items[index]))
        img = self.transform(img)
        target = 0

        return img, target