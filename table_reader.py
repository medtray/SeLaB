import torch
from torch.utils.data import Dataset
import numpy as np
from metadata import *
import json

def pad_or_crop(field, max_tokens, to_add):
    if len(field) > max_tokens:
        field = field[0:max_tokens]
    if len(field) < max_tokens:
        field+=[to_add]*(max_tokens-len(field))
    return field

class DataReaderBert(Dataset):
    def __init__(self,line,tokenizer,word2int,max_values_tokens,max_headers_tokens,max_tokens,only_values,contexts,loaded):


        self.tokenizer = tokenizer
        self.max_tokens=max_tokens
        self.padding='[PAD]'
        tokens_limit = 450

        X=[]
        y=[]

        if not loaded:
            table_cols = list(json.loads(line).values())[0]
        else:
            table_cols=line

        for index,(col, features) in enumerate(table_cols):
            values=features['content']

            np.random.shuffle(values)
            values=values[:tokens_limit]

            values_tokens = self.tokenizer.tokenize(' '.join(values), add_special_tokens=False)[:max_values_tokens]

            if only_values:
                input_tokens = ['[CLS]'] + values_tokens + ['[SEP]']+ ['[SEP]']
            else:
                #context = features['context']
                context = contexts[index]
                np.random.shuffle(context)
                context=context[:tokens_limit]
                atts_tokens = self.tokenizer.tokenize(' '.join(context), add_special_tokens=False)[:max_headers_tokens]
                input_tokens = ['[CLS]'] + values_tokens + ['[SEP]'] + atts_tokens + ['[SEP]']

            input_tokens = pad_or_crop(input_tokens, self.max_tokens, self.padding)
            word2indices = self.tokenizer.convert_tokens_to_ids(input_tokens)
            X.append(word2indices)
            y.append(word2int[col.lower()])

        attention_mask = np.where(np.array(X) != 0, 1, 0)
        X = torch.Tensor(X)
        X = X.type(dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask)
        y = torch.Tensor(y).unsqueeze(1)
        y = y.type(dtype=torch.int64)

        self.features=X
        self.labels=y
        self.masks=attention_mask


    def __getitem__(self, t):

        return self.features[t],self.masks[t], self.labels[t]
    def __len__(self):

        return len(self.features)