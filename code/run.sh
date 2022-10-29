# bert
python cloze_bert.py \
    --model_data_path /mnt/qianyuhe/model/bert-base-uncased \
    --data_path  ../Datasets/ \
    --dataset Quizzes

python cloze_bert.py \
    --model_data_path /mnt/qianyuhe/model/bert-base-uncased \
    --data_path  ../Datasets/ \
    --dataset General


python cloze_bert.py \
    --model_data_path /mnt/qianyuhe/model/bert-large-uncased \
    --data_path  ../Datasets/ \
    --dataset Quizzes

python cloze_bert.py \
    --model_data_path /mnt/qianyuhe/model/bert-large-uncased \
    --data_path  ../Datasets/ \
    --dataset General


# roberta
python cloze_roberta.py \
    --model_data_path /mnt/qianyuhe/model/roberta-base \
    --data_path  ../Datasets/ \
    --dataset Quizzes

python cloze_roberta.py \
    --model_data_path /mnt/qianyuhe/model/roberta-base \
    --data_path  ../Datasets/ \
    --dataset General


python cloze_roberta.py \
    --model_data_path /mnt/qianyuhe/model/roberta-large \
    --data_path  ../Datasets/ \
    --dataset Quizzes

python cloze_roberta.py \
    --model_data_path /mnt/qianyuhe/model/roberta-large \
    --data_path  ../Datasets/ \
    --dataset General

