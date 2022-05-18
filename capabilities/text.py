from .capability import Capability

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from copy import deepcopy
import multiprocessing
from tqdm import tqdm


# byte level tokenzier
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
            if len(buf) > max_len:
                buf = buf[:max_len]
            out.append(buf)
        return out

    def _encode_single_(self, sentence, insertable=False, add_bos=False, add_eos=False):
        if len(sentence) == 0:
            return [self.insertable]
        b = list(sentence.encode())
        if insertable:
            tmp = []
            for a in b:
                tmp.append(a)
                tmp.append(self.insertable)
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

    def _convert_to_decoder_input_and_target(self, sentence, max_len=None):
        # return input, target
        def delete_word(i, t):
            idx = random.randint(0, len(i))//2
            idx = idx*2
            random_token = random.randint(0,255)
            for idx_ in range(random.randint(1,4)):
                i.insert(idx_+idx, self.insertable)
                i.insert(idx_+idx, random_token)
                t.insert(idx_+idx, self.delete)
                t.insert(idx_+idx, random_token)
            return i, t
        def insert_word(i, t):
            idx = random.randint(2, len(i))//2
            idx=idx*2
            length = random.randint(1,len(i))*2
            if length > len(i)-2-idx:
                length = len(i)-2-idx
            for idx_ in range(idx, length, 2):
                del i[idx_:idx_+2]
                i.append(self.padding)
                i.append(self.padding)
                del t[idx_-1]
                t.append(self.padding)
            return i, t
        def random_word(i, t):
            idx = random.randint(0, len(i))//2
            length = random.randint(1, len(i))*2
            if length > len(i)-2-idx:
                length = len(i)-2-idx
            idx = idx*2
            for idx_ in range(idx, length, 2):
                random_token = random.randint(0, 255)
                i[idx_] = random_token
            return i, t
        
        i = self._encode_single_(sentence, insertable=True)
        t = deepcopy(i)
        for _ in range(max(1, random.randint(0, len(sentence)))):
            fid = random.randint(0,2)
            if fid == 0:
                i, t = insert_word(deepcopy(i), deepcopy(t))
            if fid == 1:
                i, t = random_word(deepcopy(i), deepcopy(t))
            if fid == 2:
                i, t = delete_word(deepcopy(i), deepcopy(t))
        return i, t

    def convert_to_decoder_input_and_target(self, sentences, max_len=None):
        i_list, t_list = [], []
        for sen in sentences:
            i, t = self._convert_to_decoder_input_and_target(sen, max_len)
            i_list.append(i)
            t_list.append(t)
        i_list, t_list = self._pad(i_list, max_len), self._pad(t_list, max_len)
        return i_list, t_list

    def vocab_size(self):
        return 261

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_pathes, num_references=5):
        super(TextDataset, self).__init__()
        self.data = []
        for path in file_pathes:
            with open(path, "r") as f:
                lines = f.read().split("\n")
                for i in range(0, len(lines)-num_references-1):
                    self.data.append([lines[i:i+num_references], lines[i+num_references+1]])

    def __getitem__(self, index): # returns context(list of str), next(str)
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)
        


class Text(Capability):
    def __init__(self, sia):
        super(Text, self).__init__()
        self.sia = sia
        self.tokenizer = ByteLevelTokenizer()
        self.embedding = nn.Embedding(self.tokenizer.vocab_size(), self.sia.nn.d_model)
        self.unembedding = nn.Linear(self.sia.nn.d_model, self.tokenizer.vocab_size())
        self.pe = self.sia.nn.pe
        self.logger = print
    def check_type(self, some_object):
        if type(some_object) == str:
            return True
        else:
            return False
        
    def read(self, text):
        x = self.tokenizer.encode([text], insertable=False)
        x = torch.LongTensor(x).to(self.pe.pe.device)
        x = self.embedding(x)
        x = self.pe(x)
        return x

    def write(self, memory, max_iteration=30, quality=1.0):
        # decode autoregressively
        x = [[0, 257]]
        x = torch.IntTensor(x).to(self.pe.pe.device)
        for i in range(max_iteration):
            # embedding
            x = self.embedding(x)
            x = self.pe(x)
            x, _ = self.sia.nn(x, memory)
            # unembed
            x = self.unembedding(x) # [batch_size, length, vocab_size]
            x = F.softmax(x, dim=2)
            # evaluate quality
            q = float(torch.mean(torch.max(x, dim=2).values,dim=[0, 1]))
            # predict word ID
            x = torch.argmax(x, dim=2) # [batch_size, length] IntTensor
            
            # decode
            print(x)
            x = self.tokenizer.decode(list(x))
            print(x)
            out = x
            # encode
            x = self.tokenizer.encode(x, insertable=True)
            x = torch.IntTensor(x).to(self.pe.pe.device)

            if q > quality:
                break
        return out[0]

    def setup_training(self,device, logger, text_file_pathes=[], num_references=5, batch_size=1, token_len=128, memory_size=128):
        target_modules = nn.ModuleList(
                [
                    self.sia.nn,
                    self.embedding,
                    self.unembedding
                ])
        optimizer = torch.optim.Adam(target_modules.parameters(), lr=1e-5)
        target_modules.train()
        criterion = nn.CrossEntropyLoss()
        dataset = TextDataset(text_file_pathes, num_references)
        return {
                'dataset': dataset,
                'logger': self.logger,
                'optimizer': optimizer,
                'criterion' : criterion,
                'batch_size' : batch_size,
                'token_len': 128,
                'memory_size': 128,
                'device' : device
                }
        

    def train_epoch(self, device, dataset, batch_size, logger, optimizer, criterion, token_len, memory_size):
        self.sia.nn = self.sia.nn.to(device)
        self.embedding = self.embedding.to(device)
        self.unembedding = self.unembedding.to(device)
        dataloader  = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
        
        bar = tqdm(total=len(dataset)//batch_size)
        for i, (context, next_text) in enumerate(dataloader):
            bs = len(next_text)
            # tokenize
            context = torch.IntTensor([self.tokenizer.encode(s, max_len=token_len) for s in context]).to(device)
            dec_input, dec_target = self.tokenizer.convert_to_decoder_input_and_target(next_text, max_len=token_len)
            dec_input = torch.LongTensor(dec_input).to(device)
            dec_target = torch.LongTensor(dec_target).to(device)
            # initialize optimizer
            optimizer.zero_grad()

            # initialize memory
            mem = torch.zeros(bs, memory_size, self.sia.nn.d_model).to(device)
            # encode to memory
            for c in context.split(1, dim=0):
                c = self.embedding(c[0])
                _, mem = self.sia.nn(c, mem)

            # memory only processing
            for _ in range(random.randint(0, 5)):
                mem = self.sia.nn.model(mem)

            # train decoding
            dec_input = self.embedding(dec_input)
            out, _ = self.sia.nn(dec_input, mem)
            out = self.unembedding(out)
            
            # calucate loss
            loss = criterion(torch.flatten(out, start_dim=0, end_dim=1), torch.flatten(dec_target, start_dim=0, end_dim=1))

            # backward
            loss.backward()

            # update parameters
            optimizer.step()

            bar.set_description(desc=f"batch {i}, Loss:{loss.item()}")
            bar.update(1)
            

