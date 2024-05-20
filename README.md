# Generalizing Conversational Dense Retrieval via LLM-Cognition Data Augmentation

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

## Abstract
This repository contains the source code for the ACL paper [Generalizing Conversational Dense Retrieval via LLM-Cognition Data Augmentation](https://arxiv.org/abs/2402.07092) by Chen et al. <br>

Conversational search utilizes muli-turn natural language contexts to retrieve relevant passages. Existing conversational dense retrieval models mostly view a conversation as a fixed sequence of questions and responses, overlooking the severe data sparsity problem -- that is, users can perform a conversation in various ways. Consequently, they often struggle to generalize to diverse conversations in real-world scenarios. In this work, we propose a framework for generalizing \textbf{Conv}ersational dense retrieval via LLM-cognition data \textbf{Aug}mentation (ConvAug). We first generate multi-level augmented conversations to capture the diverse nature of conversational contexts. Inspired by human cognition, we devise a cognition-aware prompting process to mitigate the generation of false positives, false negatives, and hallucinations. Moreover, we develop a difficulty-adaptive sample filter that selects challenging samples for complex conversations, thereby giving the model a larger learning space. A contrastive learning objective is then employed to train a better conversational context encoder. Extensive experiments conducted on four public datasets, under both normal and zero-shot settings, demonstrate the effectiveness, generalizability, and applicability of ConvAug.

Authors: Haonan Chen, Zhicheng Dou, Kelong Mao, Jiongnan Liu, Ziliang Zhao

## Requirements
- Python 3.10.13 <br>
- Pytorch 2.1.1+cu118 <br>
- Transformers 4.36.2 <br>
- [pytrec-eval](https://pypi.org/project/pytrec-eval/) 0.5 

## Run
- For Data Augmentation (Taking paraphrasing as an example):
```
./run_scripts/generate_data/paraphrase.sh
```
### QReCC:
- Train
```
./run_scripts/train_cdr/qrecc-conv_ance_aug.sh
```
- Test
```
./run_scripts/dense_retrieval/qrecc-conv_ance-concat.sh
```
### TopiOCQA:
- Train
```
./run_scripts/train_cdr/topiocqa-conv_ance_aug.sh
```
- Test
```
./run_scripts/dense_retrieval/topiocqa-conv_ance-concat.sh
```
### CAsT-20:
- Test
```
./run_scripts/dense_retrieval/cast20-conv_ance-concat.sh
```
### CAsT-21:
- Test
```
./run_scripts/dense_retrieval/cast21-conv_ance-concat.sh
```

## Citations
If you use the code, please cite the following paper:  
```
@article{CDMLZ2024ACL,
  author       = {Haonan Chen and
                  Zhicheng Dou and
                  Kelong Mao and
                  Jiongnan Liu and
                  Ziliang Zhao},
  title        = {Generalizing Conversational Dense Retrieval via LLM-Cognition Data
                  Augmentation},
  url          = {https://doi.org/10.48550/arXiv.2402.07092},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2402-07092.bib},
}
```
