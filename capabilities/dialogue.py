import torch
import torch.nn as nn
from .capability import Capability
import sentencepiece as spm
import datetime

class BottleneckEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, bottleneck_size=64):
        super(BottleneckEmbedding, self).__init__()
        self.f1 = nn.Embedding(vocab_size, bottleneck_size)
        self.f2 = nn.Linear(bottleneck_size, d_model)

    def forward(self, x):
        return self.f2(self.f1(x))

class Dialogue(Capability):
    def __init__(self, d_model, spm_dataset_path, spm_model_path, vocab_size=10000, bottleneck_size=64):
        super(Dialogue, self).__init__()
        self.spm_model_path = spm_model_path
        # train sentence piece
        spm.SentencePieceTrainer.Train(
            f'--input={spm_dataset_path}, --model_prefix={spm_model_path} --character_coverage=0.9995 --vocab_size={vocab_size} --pad_id=3 --control_symbols=[BOS],[EOS] --user_defined_symbols=[SEP],[BEGN_OF_METADATA],[END_OF_METADATA]'
            )
        # Load sentencepiece model
        self.sp = spm.SentencePieceProcessor(f'{self.spm_model_path}.model')

        # Initialize Embeddings
        self.embedding = BottleneckEmbedding(vocab_size, d_model, bottleneck_size)
        # Unembedding
        self.unembedding = nn.Sequential(nn.Linear(d_model, bottleneck_size), nn.Linear(bottleneck_size, vocab_size))
        # Positional Encoding

    
    @staticmethod
    def load(file_path):
        self = torch.load(file_path)
        # Load sentencepiece model
        self.sp = spm.SentencePieceProcessor(f'{self.spm_model_path}.model')
        return self

    def pad_sentences(self, sentences, pad_id=3):
        max_len = max([len(s) for s in sentences])
        ret = []
        for s in sentences:
            while len(s) < max_len:
                s.append(pad_id)
            ret.append(s)
        return ret

    def read(self, sia, memory, some): # some: list of strings
        self.sp.SetEncodeExtraOptions("bos:eos")
        id_list = [ self.sp.EncodeAsIds(s) for s in some ]
        id_list = self.pad_sentences(id_list)
        id_list = torch.LongTensor(id_list).to(memory.device)
        self.embedding.to(memory.device)
        embedded_sequece = self.embedding(id_list)
        memory, _ = sia.nn(memory, embedded_sequece)
        return memory

    def write(self, sia, memory): # decode autoreguressively
        pass

class DialogueMessage():
    def __init__(self, message='', name='Nameless', time=None):
        if time == None:
            self.time = datetime.datetime.now()
        self.name = name
        self.message.replace('[SEP]', '')
        self.message = message

    def to_str(self):
        time = self.time
        timestamp = f"{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}"
        return f"{timestamp}[SEP]{self.name}[SEP]{self.message}"

    @classmethod
    def from_str(cls, string):
        separated = string.split("[SEP]")
        timestamp = separated[0]
        name = separated[1]
        message = separated[2]

    
