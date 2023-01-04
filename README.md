# Can Pre-trained Language Models Interpret Similes as <u>Smart</u> as Human?
Code and datasets for the paper "Can Pre-trained Language Models Interpret <u>Similes</u> as Smart as Human?" (ACL 2022)

## Enviroment Details
Some main dependencies:
- python==3.6.13
- torch==1.8.1+cu111
- torchvision==0.9.1+cu111
- transformers==4.18.0
- sklearn==0.24.2
- wandb==0.12.21
- numpy==1.19.5
- pandas==1.1.5
- tqdm==4.62.3

We also provide requirements.txt. You can install the dependencies as follows:
```
conda create -n simile python=3.6
conda activate simile
pip install -r requirements.txt 
```

## Project Structure
- `Datasets/` All the data files
  - `Quizzes.csv`, `General_Corpus.csv`: We construct simile property probing datasets from both general textual corpora and human-designed questions, and the probing datasets contain 1,633 examples covering seven main categories of similes. The columns are [sentence,topic,vehicle,event,option0,option1,option2,option3,human_idx,ans_idx]. Here, `ans_idx` is the index of groundtruth.  `human_idx` is the index given by human annotators.
  - `SPGC_parsed.txt`: We collect training data from Standardized Project Gutenberg Corpus6 (SPGC). We extract similes via matching the syntactic pattern (Noun ... as ADJ as ... NOUN) and end up with 4,510 sentences. The data format is `sentence \t topic \t property \t vehicle`.
- `code` All the code files
  - `zero-shot`: The models are off-the-shelf.
  - `distant_supervision`: The models are fine-tuned with MLM objective via masking properties or our knowledge-enhanced training objective.

## How to use
### 1. Zero-shot setting
The scripts are in `code/zero-shot/run.sh`.

### 2. Training with MLM objective
```
cd code/distant_supervision
python main.py \
    --model_name_or_path bert-base-uncased \
    --dataset quiz \
    --loss_mode mlm \
    --epoch 10 \
    --lr 2e-5
```
- `loss_mode`: choose the training objective. choices=['mlm', 'ke']
- `dataset`: choose the test dataset. choices=['quiz', 'gp']


### 3. Training with KE objective
```
cd code/distant_supervision
python main.py \
    --model_name_or_path bert-base-uncased \
    --dataset quiz \
    --loss_mode ke \
    --epoch 10 \
    --lr 2e-5
```
