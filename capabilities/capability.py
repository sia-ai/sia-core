import torch
import torch.nn as nn

class Capability(nn.Module):
    def __init__(self):
        super(Capability, self).__init__()

    def check_readable_type(self, some): # returns float(0.0 to 1.0) or bool
        pass

    def read(self, some):
        pass

    def write(self, some):
        pass

    def train(self, num_epoch=1):
        pass

    def train_epoch(self, **kwargs):
        pass
    
    def save(self, file_path):
        torch.save(self, file_path)

    @staticmethod
    def load(file_path):
        return torch.load(file_path)
