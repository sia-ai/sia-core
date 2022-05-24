import torch
import torch.nn as nn

class Capability(nn.Module):
    def __init__(self):
        super(Capability, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def check_readable_type(self, some): # returns float(0.0 to 1.0) or bool
        return 0

    def read(self, sia, memory, some): # returns memory
        pass

    def write(self, sia, memory): # returns some object
        pass
    
    def save(self, file_path):
        torch.save(self, file_path)

    @staticmethod
    def load(file_path):
        return torch.load(file_path)

    @property
    def device(self):
        return self.dummy_param.device
