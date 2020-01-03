import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
from fastai.text import *

class SearchEngineDataset(Dataset):
    def __init__(self, csv_path, cols, transform=None, test=False):
        self.query_key = 'query'
        self.passage_key = 'passage'
        self.qid_key = 'qid'
        self.pid_key = 'pid'
        self.label_key = 'label'        
        self.data = pd.read_csv(csv_path, sep='\t', names=[ self.qid_key,
                                                            self.query_key, 
                                                            self.passage_key,
                                                            self.label_key,
                                                            self.pid_key
                                                            ],
                                                    dtype={self.qid_key: np.int32,
                                                            self.query_key: str, 
                                                            self.passage_key: str,
                                                            self.label_key: np.int32,
                                                            self.pid_key: np.int32}
                                                                        )
        self.data = self.data[cols]
        self.data_len = len(self.data)
        self.transform = transform
        self.test = test
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        query = row[self.query_key]
        passage = row[self.passage_key]
        if self.transform:
            query = self.transform(query)
            passage = self.transform(passage)

        x = (query, passage)

        if not self.test: 
            y = row[self.label_key]
            return x, y
        else:
            return x
      

def pad_collate(batch):

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        # Use 1 for padding, since xxpad has an index of 1
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs


    queries = [item[0][0] for item in batch]
    passages = [item[0][1] for item in batch]
    targets = torch.FloatTensor([item[1] for item in batch])
    # merge sequences
    qseqs = merge(queries)
    pseqs = merge(passages)

    return (qseqs, pseqs), targets.unsqueeze(1)


class TextTransform():
    def __init__(self, itos_path):
        self.itos = pickle.load(open(itos_path, 'rb'))
        self.base_tok = text.SpacyTokenizer('en')
        self.tokenizer = text.Tokenizer()
        self.vocab = text.Vocab(self.itos)


    def text_to_ints(self, string):
        tokens = self.tokenizer.process_text(string, self.base_tok)
        return self.vocab.numericalize(tokens)
