from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from sklearn import metrics
import transformers
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
import pdb


class SIMILE_loss_Model(nn.Module):
    def __init__(self, model_name, alpha = 10, max_seq_len = 128):
        super(SIMILE_loss_Model, self).__init__()
        self.alpha = alpha
        self.max_seq_len = max_seq_len
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def mlm_test_step(self, inputs, labels, candidates_ids, masked_index_list, ans_idx_list, device):
        bz = inputs['input_ids'].shape[0]
        for kk in inputs:
            inputs[kk] = inputs[kk].squeeze(1).to(device)
        labels = labels.squeeze(1).to(device)
        candidates_ids = [t[0].numpy() for t in candidates_ids]
        candidates_ids = torch.Tensor(candidates_ids).permute(1,0)
        

        self.model.eval()
        mlm_out = self.model(**inputs, labels=labels)

        pred_list = []
        remove_list = []
        for idx in range(bz):
            try:
                predictions_candidates = mlm_out.logits[torch.Tensor([idx]).type(torch.long), masked_index_list[idx].type(torch.long), candidates_ids[idx].type(torch.long)]
            except:
                # 存在过长[MASK]被挤出去的情况，这种时候就不要这条数据了
                remove_list.append(idx)
                continue
            answer_idx = torch.argmax(predictions_candidates).item()
            pred_list.append(answer_idx)

        
        ans_idx_list = ans_idx_list.detach().numpy()
        ans_idx_list = [ans_idx_list[i] for i in range(0, len(ans_idx_list), 1) if i not in remove_list]
        res = (ans_idx_list, pred_list)

        return res
        
    def mlm_train_mlm_step(self, inputs, labels, device):
        for kk in inputs:
            inputs[kk] = inputs[kk].squeeze(1).to(device)
        labels = labels.squeeze(1).to(device)
        # pdb.set_trace()
       
        outputs = self.model(**inputs, labels=labels)        
        return outputs.loss
    
    def mlm_train_ke_step(self, inputs, labels, top_idx_list, veh_idx_list, masked_index, device):
        for kk in inputs:
            inputs[kk] = inputs[kk].squeeze(1).to(device)
        labels = labels.squeeze(1).to(device)
        # pdb.set_trace()
       
        mlm_out = self.model(**inputs, labels=labels, output_hidden_states=True)        
        

        # keloss
        top_idx_list = self.permute_list(top_idx_list)
        veh_idx_list = self.permute_list(veh_idx_list)
        masked_index = masked_index.detach().numpy()
        # pdb.set_trace()
        
        mse_a = None
        mse_b = None
        pad_mark = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        bz = inputs['input_ids'].shape[0]
        for ii in range(bz):
            vec1 = None
            vec2 = None
            vec3 = None

            if masked_index[ii] >= self.max_seq_len:
                continue

            _ = []
            for jj in range(len(top_idx_list[ii])):
                if top_idx_list[ii][jj] != pad_mark and int(top_idx_list[ii][jj]) < self.max_seq_len:
                    _.append(mlm_out.hidden_states[-1][ii][int(top_idx_list[ii][jj])])

            if len(_) == 0:
                continue
            vec1 = torch.mean(torch.stack(_), 0)
            vec2 = mlm_out.hidden_states[-1][ii][masked_index[ii]]
            
            _ = []
            for jj in range(len(veh_idx_list[ii])):
                if veh_idx_list[ii][jj] != pad_mark and int(veh_idx_list[ii][jj]) < self.max_seq_len:
                    _.append(mlm_out.hidden_states[-1][ii][int(veh_idx_list[ii][jj])])
            if len(_) == 0:
                continue
            vec3 = torch.mean(torch.stack(_), 0)
            
            if mse_a == None:
                mse_a = (vec1 + vec2).reshape(1,vec3.size()[0])
            else:
                mse_a = torch.cat((mse_a, (vec1 + vec2).reshape(1,vec3.size()[0])), 0)
            
            if mse_b == None:
                mse_b = vec3.reshape(1,vec3.size()[0])
            else:
                mse_b = torch.cat((mse_b, vec3.reshape(1,vec3.size()[0])), 0)

        loss_fct = MSELoss()
        ke_loss = loss_fct(mse_a, mse_b)
        all_loss = self.alpha * ke_loss + mlm_out.loss

        return all_loss
    

    def permute_list(self, tmp):
        tmp = [t.numpy() for t in tmp]
        tmp = torch.Tensor(tmp).permute(1,0).detach().numpy()
        return tmp
