import torch
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import random
import os
import argparse
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 设置 cpu 的随机数种子
    torch.cuda.manual_seed(seed) # 对于单张显卡，设置 gpu 的随机数种子
    torch.cuda.manual_seed_all(seed) # 对于多张显卡，设置所有 gpu 的随机数种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cloze_test(data):
    text_list = []
    candidates_list = []
    gt_list = []
    cate_list = []

    for index,row in data.iterrows():
        text_list.append(row["sentence"].lower())
        candidates_list.append([row["option0"],row["option1"],row["option2"],row["option3"]])
        gt_list.append(int(row["ans_idx"]))

    gt_list_match = []
    pred_list = []
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    for ii in tqdm(range(len(text_list))):
        text = text_list[ii]
        text = text.replace('_','[MASK]')
        candidates = candidates_list[ii]
        
        indexed_tokens = tokenizer(text)['input_ids']
        tokens_tensor = torch.tensor([indexed_tokens]).to("cuda")
        masked_index = indexed_tokens.index(mask_id)

        cvt_candidates = []
        for cc in candidates:
            cvt_candidates.append(cc)
        candidates_ids = tokenizer.convert_tokens_to_ids(cvt_candidates)

        language_model.eval()
        predictions = language_model(tokens_tensor)
        predictions_candidates = predictions.logits[0, masked_index, candidates_ids]
        answer_idx = torch.argmax(predictions_candidates).item()

        pred_list.append(answer_idx)
        gt_list_match.append(gt_list[ii])
    return  pred_list, gt_list_match, cate_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Quizzes', choices=['Quizzes', 'General'])
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--model_data_path', default='bert-base-uncased', type=str)
    parser.add_argument('--data_path', default='../Datasets/', type=str)
    arg = parser.parse_args()

    if arg.dataset == 'Quizzes':
         file_name = arg.data_path + 'Quizzes'
    else:
        file_name = arg.data_path + 'General_Corpus'
    data = pd.read_csv(file_name + '.csv')

    tokenizer = BertTokenizer.from_pretrained(arg.model_data_path)
    language_model = BertForMaskedLM.from_pretrained(arg.model_data_path)
    language_model.to('cuda')

    set_random_seed(arg.seed)

    pred_list, gt_list_match, cate_list = cloze_test(data)
    print(round(metrics.accuracy_score(gt_list_match,pred_list)*100,2))