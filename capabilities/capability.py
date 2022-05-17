import torch
import torch.nn as nn
from tqdm import tqdm
    
class Capability:
    def __init__(self, sia=None):
        self.logger = nn.Identity()
        self.sia = sia

    def check_type(self, some_object):
        return 0.

    def read(self):
        pass

    def write(self):
        pass

    def train(self, num_epoch, device=None, logger=tqdm.write, **kwargs):
        if not device:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger("Setup training...")
        settings = self.setup_training(device = device, logger=logger, **kwargs)
        logger("Setup Complete!")
        for epoch in tqdm(range(num_epoch)):
            self.train_epoch(**settings)

    def setup_training(self):
        pass

    def train_epoch(self, dataset=None, batch_size=1, criterion=nn.Identity(), optimizer=None):
        pass
