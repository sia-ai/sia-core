from capabilities import Capability

import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteLevelTokenizer():
    def __init__(self, encoding='utf-8', padding=256, insertable=257, bos=258, eos=259, delete=260):
        self.encoding = encoding
        self.padding=padding
        self.insertable=insertable
        self.bos=bos
        self.eos=eos
        self.delete=delete

    # x = [[int, int,...][int, int,...]...]
    def _pad(self, x, max_len=None):
        if not max_len:
            max_len = max([len(s) for s in x])
        out = []
        for s in x:
            buf = s
            while len(buf) < max_len:
                buf.append(self.padding)
            if len(buf) < max_len:
                buf = buf[:max_len-1]
            out.append(buf)
        return out

    def _encode_single_(self, sentence, insertable=False, add_bos=False, add_eos=False):
        b = list(sentence.encode())
        if insertable:
            tmp = []
            for a in b:
                tmp.append(b)
            b = tmp
        if add_bos:
            b = self.bos + b
        if add_eos:
            b = b = self.eos
        return b

    def _decode_single_(self, byte_seq):
        # if detected eos, finish decode forced.
        # padding or insertable will be interpreted empty.
        # if detected delete, delete it's left side byte.
        res_bytes = []
        for b in byte_seq:
            if b == self.eos:
                break
            if b == self.delete:
                res_bytes = res_bytes[:-1] # like backspace
            if b <= 255:
                res_bytes.append(b)
        return bytes(res_bytes).decode(self.encoding, errors='replace')

    def encode(self, sentence, max_len=None, insertable=True):
        if type(sentence) == str:
            return self._pad(sentence, max_len=max_len)
        mat = [self._encode_single_(s, insertable=insertable) for s in sentence]
        return self._pad(mat, max_len)

    def decode(self, sequence):
        if type(sequence[0]) == int:
            return self._decode_single_(sequence)
        output_list = []
        for s in sequence:
            output_list.append(self._decode_single_(s))
        return output_list

    def vocab_size(self):
        return 261


class Text(Capability):
    def __init__(self):
        super(Text, self).__init__()
        self.tokenizer = ByteLevelTokenizer()
        self.embedding = nn.Embedding(self.tokenizer.vocab_size(), self.sia.nn.d_model)
        self.unembedding = nn.Linear(self.sia.nn.d_model, self.tokenizer.vocab_size())
        self.pe = self.sia.nn.pe
    def check_type(self, some_object):
        if type(some_object) == str:
            return True
        else:
            return False
        
    def read(self, text):
        x = self.tokenizer.encode(text, insertable=False)
        x = self.embedding(x)
        x = self.pe(x)
        return x

    def write(self, memory, max_iteration=30, quality=1.0):
        # decode autoregressively
        x = self.tokenizer.encode('')
        x = self.embedding(x)
        x = self.pe(x)
        for i in range(max_iteration):
            out, _ = self.sia.nn(x, memory)
            # unembed
            out = self.unembedding(out) # [batch_size, length, vocab_size]
            out = F.softamx(out, dim=2)
            # evaluate quality
            q = float(torch.mean(torch.max(out, dim=2),dim=[0, 1]))
            # predict word ID
            x = torch.argmax(out, dim=2) # [batch_size, length] IntTensor
            if q > quality:
                break
        output = self.tokenizer.decode(x)
        return output

    def setup_train(self, text_file_path=):
        pass

    def train_epoch(self, device, dataset,  logger, optimizer, criterion):
        logger = 
