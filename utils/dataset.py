import pandas as pd
import base64

from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset


class AttractionsDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 label_maps,
                 transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.label_maps = label_maps
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img = Image.open(BytesIO(base64.b64decode(self.img_labels.iloc[idx]['img'])))
        lbl = self.label_maps[self.img_labels.iloc[idx]['name']]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return img, lbl
