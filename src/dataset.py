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
                 hparam_search: bool= False, seed: int = 42):
        self.img_dir = os.path.join(root_dir, "imgs")
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        if hparam_search:
            # only train on 1/4 of the dataset
            reduced_dataset_size = len(os.listdir(self.img_dir))//4
            self.images = os.listdir(self.img_dir)[:reduced_dataset_size]
        else:
            self.images = os.listdir(self.img_dir)
        labels = pd.read_csv(os.path.join(root_dir, "car_imgs_4000.csv"))
        self.labels = {row["filename"]: [row["perspective_score_hood"], row["perspective_score_backdoor_left"]] for idx, row in labels.iterrows()}

        # split the data into 10 folds (9 for train/val and 1 for test)
        skf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        splits = skf.split(list(range(len(self.images))))
        train_val_idx, test_idx = list(splits)[fold]

        train_val_imgs = [self.images[i] for i in train_val_idx]
        test_imgs = [self.images[i] for i in test_idx]

        # split train/val-split into 9 folds (8 for train and 1 for val)
        skf2 = KFold(n_splits=num_folds-1, random_state=seed, shuffle=True)
        splits2 = skf2.split(list(range(len(train_val_idx))))
        train_idx, val_idx = list(splits2)[0]

        if split == "train":
            self.images = [train_val_imgs[i] for i in train_idx]
        elif split == "val":
            self.images = [train_val_imgs[i] for i in val_idx]
        elif split == "test":
            self.images = test_imgs
        else:
            raise ValueError(f"Invalid split: {split}")


    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(image_path)
        image = self.transform(image)
        label = torch.tensor(self.labels[self.images[idx]])

        return image, label


def get_dataloaders(cfg: DictConfig, train_transform: transforms.Compose = None, val_transform: transforms.Compose = None, fold: int = 0, search: bool=False):
    train_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, split="train", transform=train_transform, fold=fold, hparam_search=search)
    val_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, split="val", transform=val_transform, fold=fold, hparam_search=search)
    test_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, split="test", transform=val_transform, fold=fold)

    for test_file in test_dataset.images:
        assert test_file not in train_dataset.images, f"Test file{test_file} in train set"
        assert test_file not in val_dataset.images, f"Test file{test_file} in val set"
    for val_file in val_dataset.images:
        assert val_file not in train_dataset.images, f"Val file{val_file} in train set"

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