import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from PIL import Image
import pandas as pd

import os
from omegaconf import DictConfig, OmegaConf

from utils import str_to_class


class CarImageDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose = None,
                 split: str = "train", num_folds: int = 10, fold: int = 0,
                 seed: int = 42):
        self.img_dir = os.path.join(root_dir, "imgs")
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.images = os.listdir(self.img_dir)
        labels = pd.read_csv(os.path.join(root_dir, "car_imgs_4000.csv"))
        self.labels = {row["filename"]: [row["perspective_score_hood"], row["perspective_score_backdoor_left"]] for idx, row in labels.iterrows()}

        skf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        splits = skf.split(list(range(len(self.images))))
        train_val_idx, test_idx = list(splits)[fold]

        skf2 = KFold(n_splits=num_folds-1, random_state=seed, shuffle=True)
        splits2 = skf2.split(list(range(len(train_val_idx))))
        train_idx, val_idx = list(splits2)[0]

        if split == "train":
            self.images = [self.images[i] for i in train_idx]
        elif split == "val":
            self.images = [self.images[i] for i in val_idx]
        elif split == "test":
            self.images = [self.images[i] for i in test_idx]


    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(image_path)
        image = self.transform(image)
        label = torch.tensor(self.labels[self.images[idx]])

        return image, label


def get_dataloaders(cfg: DictConfig, fold: int):
    train_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, split="train", fold=fold)
    val_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, split="val", fold=fold)
    test_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, split="test", fold=fold)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hparams.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    dataloaders =  get_dataloaders(cfg=cfg, fold=0)

    for batch in dataloaders["train"]:
        print(batch)
        break
    # print(dataset[0])