import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from sklearn import model_selection,metrics
import random
import wandb
import os
import pdb
import argparse
import numpy as np
import pandas as pd
from model import SIMILE_loss_Model
from dataloader import DatasetforSimile, DatasetforSimileTest


def read_train_data(data_dir,max_row = None):
    dataset = []
    with open(data_dir, 'r', encoding="utf-8") as f:
        idx = 0
        for data in f.readlines():
            if max_row != None:
                idx += 1
                if idx > max_row:
                    break
            sentence, topic, property, vehicle = data.replace('\n','').split('\t')
            dataset.append([sentence, topic, property, vehicle ])
    f.close()
    return dataset


def read_test_data(data_dir,max_row = None):
    data = pd.read_csv(data_dir)
    data = data.sample(frac=1)
    dataset = []     
    if max_row != None:
        try:
            for index,row in data.iterrows():
                dataset.append([row["sentence"], row["option0"],row["option1"],row["option2"],row["option3"],row["ans_idx"],row["topic"],row["option"+str(row["ans_idx"])],row["vehicle"]])
        except:
            for index,row in data.iterrows():
                dataset.append([row["old_sentence"], '-' , '-' , '-' , '-' ,0,'-',row["property"],'-'])
    else:
        try:
            for index,row in data[:max_row].iterrows():
                dataset.append([row["sentence"], row["option0"],row["option1"],row["option2"],row["option3"],row["ans_idx"],row["topic"],row["option"+str(row["ans_idx"])],row["vehicle"]])
        except:
            for index,row in data[:max_row].iterrows():
                dataset.append([row["old_sentence"], '-' , '-' , '-' , '-' ,0,'-',row["property"],'-'])
        
    return dataset
    

def train_loop(dataloader, model,  optimizer, lr_scheduler, epoch, total_loss, loss_mode):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, sample in enumerate(dataloader, start=1):
        # return: inputs(mask后), labels, top_idx_list, veh_idx_list
        inputs = sample[0]
        labels = sample[1]
        top_idx_list = sample[2]
        veh_idx_list = sample[3]
        masked_index = sample[4]

        if loss_mode == 'mlm':
            loss = model.mlm_train_mlm_step(inputs, labels, device)
        elif loss_mode == 'ke':
            loss = model.mlm_train_ke_step(inputs, labels, top_idx_list, veh_idx_list, masked_index, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss, total_loss/(finish_batch_num + batch)


def test_loop(validation_dataloader, model):
    ans_list = []
    pred_list = []
    final_list = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(validation_dataloader):
            inputs = sample[0]
            labels = sample[1]
            candidates_ids = sample[2]
            masked_index_list = sample[3]
            ans_idx_list = sample[4]
            
            res = model.mlm_test_step(inputs, labels, candidates_ids, masked_index_list, ans_idx_list, device)
            ans_list += res[0]
            pred_list += res[1]

        acc = round(metrics.accuracy_score(ans_list,pred_list)*100,2)

    return round(metrics.accuracy_score(ans_list,pred_list)*100,2)


if __name__ == "__main__":
    print("start!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--description', type=str, default='bert-base')
    parser.add_argument('--seed', type=int, default=234)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--save_weight', default=False, action='store_true')
    parser.add_argument("--save_folder_name", default="ckp/", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--train_data_path", default="../../Datasets/SPGC_parsed.txt", type=str)
    parser.add_argument('--dataset', type=str, default='quiz', choices=['quiz', 'gp'])
    parser.add_argument('--loss_mode', type=str, default='ke', choices=['mlm', 'ke'])



    arg = parser.parse_args()

    if arg.dataset == 'quiz':
        arg.test_data_path = '../../Datasets/Quizzes.csv'
    elif arg.dataset == 'gp':
        arg.test_data_path = '../../Datasets/General_Corpus.csv'

    if arg.wandb:
        wandb.init(
            project="simile",
            name=arg.description,
        )

    # 初始化seed
    seed = arg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model = SIMILE_loss_Model(arg.model_name_or_path, alpha = arg.alpha).to(device)
    

    if 'roberta' in arg.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(arg.model_name_or_path, model_max_length = arg.max_length, add_prefix_space=True)
        arg.mode = 'roberta'
    elif 'bert' in arg.model_name_or_path:
        arg.mode = 'bert'
        tokenizer = AutoTokenizer.from_pretrained(arg.model_name_or_path, model_max_length = arg.max_length)
    vocab_size = len(tokenizer)

    # 加载数据集
    train_set = read_train_data(arg.train_data_path)
    train_dataset = DatasetforSimile(train_set,tokenizer, max_seq_len = arg.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size = arg.batch_size, shuffle = True)
    test_set = read_test_data(arg.test_data_path)
    test_dataset = DatasetforSimileTest(test_set,tokenizer, arg.mode, max_seq_len = arg.max_length)
    test_loader = DataLoader(test_dataset, batch_size = arg.batch_size, shuffle = False)



    # 优化器
    optimizer = AdamW(model.parameters(), lr=arg.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=arg.epoch*len(train_dataloader),
    )

    total_loss = 0.
    best_acc = 0.
    for t in range(arg.epoch):
        print(f"Epoch {t+1}/{arg.epoch}\n-------------------------------")
        total_loss, average_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss, arg.loss_mode)
        print('average_loss:')
        print(average_loss)

        final_metric = test_loop(test_loader, model)
        print(final_metric)

        if arg.save_weight:
            if not os.path.exists(arg.save_folder_name):
                os.makedirs(arg.save_folder_name)
            
            if final_metric > best_acc:
                best_acc = final_metric
            print('saving new weights...\n')
            torch.save(
                model.state_dict(), 
                'arg.save_folder_name' + f'epoch_{t+1}_test_acc_{(100*best_acc):0.3f}_weights.bin'
            )

        if arg.wandb: wandb.log({
            'acc':best_acc
        })
        torch.cuda.empty_cache()


