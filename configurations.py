import torch

from src.loss.DiceLoss import DiceLoss


class Config:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 100
        self.lr = 1e-4
        self.weight_decay = 0.00005
        self.dev = "cpu"
        self.device = torch.device(self.dev)
        self.momentum_sgd = 0.9
        self.datapath = '/Users/bob/PycharmProjects/UAV-2023/dataset/'
        self.criterion = None
        self.optimizer = None
        self.scaler = None
        self.scheduler = None

    def set_model_related_configs(self, model):
        # use DiceLoss for UNET
        # self.criterion = DiceLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2, eta_min=5e-5)