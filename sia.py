from models.lead import LEADForSIA, LEAD
import torch

class SIASession():
    def __init__(self, sia, memory):
        self.sia = sia
        self.memory = memory
    def __call__(self, some_object):
        return self

    def get(self, capability):
        pass

    def close(self):
        del self
    
class SIA():
    def __init__(self, d_model=256):
        self.capabilities = []
        self.nn = LEADForSIA(d_model, LEAD(d_model))

    def open_session(self, num_memory_token=16):
        return SIASession(self, self.nn.allocate(num_memory_token))
    
    def add_capability(self, capability_class):
        self.capabilities.append(capability_class(sia=self))
    



# tests

from capabilities.text import Text
import os
if os.path.exists('sia.pt'):
    sia = torch.load('sia.pt')
else:
    sia = SIA()
sia.add_capability(Text)
tc = sia.capabilities[0]
tc.train(1, text_file_pathes=["./dialogue_short.txt"], logger=print, batch_size=30, token_len=64)
#sess = sia.open_session()
torch.save(sia, 'sia.pt')
