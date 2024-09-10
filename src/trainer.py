import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf, DictConfig
import logging
from tqdm import tqdm

from dataset import get_dataloaders
from utils import str_to_class


class Trainer:
    def __init__(self, cfg: DictConfig, dataloaders: dict[DataLoader]) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level=logging.INFO)

        self._cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = str_to_class(cfg.model.name)(**cfg.model.params)
        self.model.to(self.device)
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.test_dataloader = dataloaders["test"]
        self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.params)
        self.optimizer = str_to_class(cfg.hparams.optimizer.name)(self.model.parameters(), **cfg.hparams.optimizer.params)
        # self.scheduler = str_to_class(cfg.hparams.scheduler.name)(self.optimizer, **cfg.hparams.scheduler.in_params)

        self.logger.info(f"Device: {self.device}")
        
        
    def train(self):
        self.model.train()
        self.logger.info("Starting training...")
        for epoch in range(self._cfg.hparams.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self._cfg.hparams.epochs}")
            for i, (X, y) in enumerate(tqdm(self.train_dataloader)):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.val()

        self.save_model()
        self.logger.info("Training completed!")

    def val(self):
        self.model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(self.val_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

    def save_model(self):
        torch.save(self.model.state_dict(), self._cfg.model.save_path)

    
if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    for fold in range(cfg.dataset.params.num_folds-9):
        dataloaders = get_dataloaders(cfg=cfg, fold=fold)
        trainer = Trainer(cfg=cfg, dataloaders=dataloaders)
        trainer.train()
