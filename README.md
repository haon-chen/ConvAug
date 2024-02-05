# GOLD

This anonymous repository contains the source code for the ARR submission "Generalizing Conversational Context Encoder via LLM-Cognition Data Augmentation".

To ensure anonymity, we will cite the source of some codes later.

## Requirements
- Python 3.10.13 <br>
- Pytorch 2.1.1+cu118 <br>
- Transformers 4.36.2 <br>
- [pytrec-eval](https://pypi.org/project/pytrec-eval/) 0.5  

## Usage
- data 
  - We provide the code for paraphrasing conversations as an example in ./data_augmentation
  - To ensure anonymity, we will provide the source of the original datasets later.

## Run
- For Data Augmentation (Taking paraphrasing as an example):
```
./run_scripts/generate_data/paraphrase.sh
```
- For QReCC:
- Train
```
./run_scripts/train_cdr/qrecc-conv_ance_aug.sh
```
- Test
```
./run_scripts/dense_retrieval/qrecc-conv_ance-concat_aug.sh
```
- For TopiOCQA:
- Train
```
./run_scripts/train_cdr/topiocqa-conv_ance_aug.sh
```
- Test
```
./run_scripts/dense_retrieval/topiocqa-conv_ance-concat.sh
```
- For CAsT-20:
- Test
```
./run_scripts/dense_retrieval/cast20-conv_ance-concat.sh
```
- For CAsT-21:
- Test
```
./run_scripts/dense_retrieval/cast21-conv_ance-concat.sh
```