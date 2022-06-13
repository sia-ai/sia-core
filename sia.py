import torch

from models.lead import LEADLayerStack, LEADMemoryWrapper

class SIASession():
    def __init__(self, sia, memory):
        self.sia = sia
        self.memory = memory

    def read(self, some_object):
        pass

    def write(self, capability):
        pass

class SIA():
    def __init__(self, d_model=512):
        lead = LEADLayerStack(d_model)
        self.nn = LEADMemoryWrapper(d_model, lead)

    def save(self, model_path):
        torch.save(self.nn, model_path)

    @staticmethod
    def load(model_path, **kwargs):
        nn = torch.load(model_path, **kwargs)
        self = SIA()
        self.nn = nn
        return self

    def to(self, device):
        self.nn.to(device)
