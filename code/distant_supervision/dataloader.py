
import torch
import re
import pickle
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random


class DatasetforSimile(Dataset):
    def __init__(self, data, tokenizer,  max_seq_len = 128, max_comp_len = 5):
        super(DatasetforSimile, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_comp_len = max_comp_len
        
        
    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx): 
        sample = self.data[idx]
        sent = sample[0]
        property = sample[2]
  
        topic = sample[1]
        vehicle = sample[3]
        
        masked_sent = sent.replace('_',self.tokenizer.mask_token)
        origin_sent = sent.replace('_',property)
        
        inputs = self.tokenizer(masked_sent, max_length = self.max_seq_len, padding = 'max_length', truncation = True, return_tensors = 'pt')
        labels = self.tokenizer(origin_sent, max_length = self.max_seq_len, padding = 'max_length', truncation = True, return_tensors = 'pt')["input_ids"]
        labels = torch.where(inputs.input_ids == self.tokenizer.mask_token_id, labels, -100)
        masked_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        _ = sent.split(' ')
        if _[0] != topic:
            topic_tok = self.tokenizer.encode(' ' + topic)[1:-1]
        else:
            topic_tok = self.tokenizer.encode(topic)[1:-1]
        property_tok = self.tokenizer.encode(' ' + property)[1:-1]
        vehicle_tok = self.tokenizer.encode(' ' + vehicle)[1:-1]

        top_idx_list = self.find_idx(topic_tok, inputs['input_ids'][0].tolist())
        veh_idx_list = self.find_idx(vehicle_tok, inputs['input_ids'][0].tolist())
        pad_mark = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        # 把top_idx_list和veh_idx_list给pad/truncation到一样的长度
        if len(top_idx_list) <= self.max_comp_len:
            top_idx_list.extend(
            [pad_mark for i in range(self.max_comp_len - len(top_idx_list))])
        if len(veh_idx_list) <= self.max_comp_len:
            veh_idx_list.extend(
            [pad_mark for i in range(self.max_comp_len - len(veh_idx_list))])
        
        top_idx_list = top_idx_list[:self.max_comp_len]
        veh_idx_list = veh_idx_list[:self.max_comp_len]
        # pdb.set_trace()
        return inputs, labels, top_idx_list, veh_idx_list, masked_index
    
    def find_idx(self, tok_list, indexed_tokens):
        idx_list = []
        for ii,tok in enumerate(indexed_tokens):
            flag = 1
            for jj in range(len(tok_list)):
                if (ii + jj) >= len(indexed_tokens):
                    flag = 0
                    break
                if tok_list[jj] == indexed_tokens[ii + jj]:
                    idx_list.append(ii + jj)
                else:
                    flag = 0
            if flag == 1:
                break
        return idx_list


class DatasetforSimileTest(Dataset):
    def __init__(self, data, tokenizer, mode, max_seq_len = 128, max_comp_len = 5):
        super(DatasetforSimileTest, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_comp_len = max_comp_len
        self.mode = mode
        
    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx): 
        
        sample = self.data[idx]
        sent = sample[0]
        candidates_list = [sample[1],sample[2],sample[3],sample[4]]
        ans_idx = sample[5]
        property = sample[7]

        masked_sent = sent.replace('_',self.tokenizer.mask_token)
        origin_sent = sent.replace('_',property)
        
        inputs = self.tokenizer(masked_sent, max_length = self.max_seq_len, padding = 'max_length', truncation = True, return_tensors = 'pt')
        labels = self.tokenizer(origin_sent, max_length = self.max_seq_len, padding = 'max_length', truncation = True, return_tensors = 'pt')["input_ids"]
        labels = torch.where(inputs.input_ids == self.tokenizer.mask_token_id, labels, -100)
        masked_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        # 转换选项
        cvt_candidates = []
        for cc in candidates_list:
            if cc != '$':
                cvt_candidates.append(cc)

        candidates_ids = self.tokenizer(cvt_candidates,  add_special_tokens=False)['input_ids']
        
        return inputs, labels, candidates_ids, masked_index[0], ans_idx
    
