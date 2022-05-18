from models.lead import LEADForSIA, LEAD
import torch

class SIASession():
    def __init__(self, sia, memory):
        self.sia = sia
        self.memory = memory

    def __call__(self, some_object):
        # select capability
        now_capa = None
        now_pirority = 0
        for c in self.sia.capabilities:
            p = c.check_type(some_object)
            if p == True:
                p = 1
            if p == 0 or p == 0. or p == False:
                continue
            if now_pirority < p:
                now_pirority = p
                now_capa = c
        if now_capa == None:
            raise f"{type(some_object)} is not readable"
        capa = now_capa
        t = capa.read(some_object)
        _, mem = self.sia.nn(t, self.memory)
        return self
        
    def get(self, capability_class, **kwargs):
        capa = self.sia.get_capability(capability_class)
        if capa == None:
            raise f"capability {capability_class.__name__} is not found"
        out = capa.write(self.memory, **kwargs)
        return out

    def close(self):
        del self
        return None
    
class SIA():
    def __init__(self, d_model=256):
        self.capabilities = []
        self.nn = LEADForSIA(d_model, LEAD(d_model))

    def open_session(self, num_memory_token=16):
        return SIASession(self, self.nn.allocate(num_memory_token))
    
    def add_capability(self, capability_class):
        self.capabilities.append(capability_class(sia=self))

    def get_capability(self ,capaility_class):
        for capa in self.capabilities:
            if type(capa) == capaility_class:
                return capa
        return None
    

# tests

from capabilities.text import Text
import os
import datetime
import re
if os.path.exists('sia.pt'):
    sia = torch.load('sia.pt')
else:
    sia = SIA()

sia.add_capability(Text)

train_mode=True

if train_mode:
    tc = sia.capabilities[0]
    tc.train(5, text_file_pathes=["./dialogue_short.txt"], logger=print, batch_size=300, token_len=64,num_references=2)
    torch.save(sia, 'sia.pt')

else:
    user_name = os.environ["USER"]
    sess = sia.open_session()
    while True:
        i = input(f"{user_name} >")
        t = datetime.datetime.now()
        ts = f"{t.year}_{t.month}_{t.day}_{t.hour}_{t.minute}"
        sia_prompt = f"{ts}[SEP]{user_name}[SEP]{i}"
        output_text = sess(sia_prompt).get(Text)
        print(output_text)


