# bert
python cloze_bert.py \
    --model_data_path bert-base-uncased \
    --data_path  ../../Datasets/ \
    --dataset Quizzes

python cloze_bert.py \
    --model_data_path bert-base-uncased \
    --data_path  ../../Datasets/ \
    --dataset General


python cloze_bert.py \
    --model_data_path bert-large-uncased \
    --data_path  ../../Datasets/ \
    --dataset Quizzes

python cloze_bert.py \
    --model_data_path bert-large-uncased \
    --data_path  ../../Datasets/ \
    --dataset General


# roberta
python cloze_roberta.py \
    --model_data_path roberta-base \
    --data_path  ../../Datasets/ \
    --dataset Quizzes

python cloze_roberta.py \
    --model_data_path roberta-base \
    --data_path  ../../Datasets/ \
    --dataset General


python cloze_roberta.py \
    --model_data_path roberta-large \
    --data_path  ../../Datasets/ \
    --dataset Quizzes

python cloze_roberta.py \
    --model_data_path roberta-large \
    --data_path  ../../Datasets/ \
    --dataset General

