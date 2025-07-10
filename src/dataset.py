import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image


class ImageDataset(Dataset):
    def __init__(self, manifest, img_dir, mask_dir, transform=None, target_transform=None):
        self.manifest = pd.read_csv(manifest)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        img_path = self.manifest.loc[idx, "img"]
        mask_path = self.manifest.loc[idx, "mask"]
        img = decode_image(img_path)
        mask = decode_image(mask_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.transform(mask)
        return img, mask
