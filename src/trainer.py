import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

import os
from omegaconf import OmegaConf, DictConfig
import logging
from tqdm import tqdm

from dataset import get_dataloaders
from utils import str_to_class


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(self, cfg: DictConfig, dataloaders: dict[DataLoader]) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level=logging.INFO)

        self._cfg = cfg
        if not os.path.exists(cfg.model.save_dir):
            os.makedirs(cfg.model.save_dir)
        id = len(os.listdir(cfg.model.save_dir))
        self.save_path = os.path.join(cfg.model.save_dir, f"model_{id}.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = str_to_class(cfg.model.name)(**cfg.model.params)
        self.model.to(self.device)
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.test_dataloader = dataloaders["test"]
        self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.params)
        self.optimizer = str_to_class(cfg.hparams.optimizer.name)(self.model.parameters(), **cfg.hparams.optimizer.params)
        self.early_stopper = EarlyStopper(**cfg.hparams.early_stopper.params)
        self.logger.info(f"Device: {self.device}")
        
        
    def train(self,save_model: bool = False, verbose: bool = False):
        self.model.train()
        self.logger.info("Starting training...")
        for epoch in range(self._cfg.hparams.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self._cfg.hparams.epochs}")
            running_loss = 0.0
            for i, (X, y) in enumerate(tqdm(self.train_dataloader)):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()



            val_loss = self.val()
            if verbose:
                self.logger.info(f"Epoch {epoch+1} training loss: {running_loss/len(self.train_dataloader)}")
                self.logger.info(f"Epoch {epoch+1} validation loss: {val_loss}")
            if self.early_stopper.early_stop(val_loss):
                self.logger.info("Early stopping...")
                break
        if save_model:
            self.save_model()
        self.logger.info("Training completed!")
        return {"best_val_loss": self.early_stopper.min_validation_loss, "last_val_loss": val_loss}

    def val(self):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (X, y) in enumerate(self.val_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                running_loss += loss.item()
            return running_loss/len(self.val_dataloader)

    def test(self, model_path: str, verbose: bool = False):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            pred_hood = []
            pred_backdoor = []
            true_hood = []
            true_backdoor = []
            for i, (X, y) in enumerate(self.test_dataloader):
                true_hood.append(y[0,0].item())
                true_backdoor.append(y[0,1].item())
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                running_loss += loss.item()
                pred_hood.append(pred.cpu()[0,0].item())
                pred_backdoor.append(pred.cpu()[0,1].item())
                pred_dict = {
                    "Predicted visibility hood": pred.cpu()[0,0].item(),
                    "Predicted visibility left back door": pred.cpu()[0,1].item()
                }
                true_dict = {
                    "True visibility hood": y.cpu()[0,0].item(),
                    "True visibility left back door": y.cpu()[0,1].item()
                }
                if verbose:
                    self.logger.info(f"Predicted visibility: {pred_dict}")
                    self.logger.info(f"True visibility: {true_dict}")
                    self.logger.info(f"Test loss: {loss.item()}")

        r2_hood = r2_score(true_hood, pred_hood)
        r2_backdoor = r2_score(true_backdoor, pred_backdoor)
        mse_hood = mean_squared_error(true_hood, pred_hood)
        mse_backdoor = mean_squared_error(true_backdoor, pred_backdoor)
        average_test_loss = running_loss/len(self.test_dataloader)
        self.logger.info(f"R2 score for hood: {r2_hood}")
        self.logger.info(f"R2 score for backdoor: {r2_backdoor}")
        self.logger.info(f"MSE for hood: {mse_hood}")
        self.logger.info(f"MSE for backdoor: {mse_backdoor}")
        self.logger.info(f"Average test MSE: {average_test_loss}")

        return {"MSE_hood": mse_hood, "MSE_backdoor": mse_backdoor, "r2_hood": r2_hood, "r2_backdoor": r2_backdoor}

    def save_model(self):
        torch.save(self.model.state_dict(), self._cfg.model.save_path)

    
if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    for fold in range(cfg.dataset.params.num_folds-9):
        dataloaders = get_dataloaders(cfg=cfg, fold=fold)
        trainer = Trainer(cfg=cfg, dataloaders=dataloaders)
        trainer.train()
