# Intro
This is the repo of for EMNLP 2021 paper "Progressive Adversarial Learning for Bootstrapping: A Case Study on Entity Set Expansion"

# Usage

## Dataset
CoNLL and OntoNotes datasets can be downloaded from [here](https://drive.google.com/file/d/102bhotYDMPlU0ojsmFXLmQttighNP9DR/view?usp=sharing); External pre-training datasets can be downloaded from [here](https://drive.google.com/file/d/1CulQu5oixrhBev4ECFhTQARiaCghwHtM/view?usp=sharing).

After downloading, please unarchive them and put them into "dataset" folder at the root directory.

## Pre-training
Using self-supervised and supervised pre-training as

```bash
python -u pretrain_self.py --output_model_file models/pretrain_self_100_local --device 0 --local > logs/pretrain_self_100_local.txt
python -u pretrain_sup.py --input_model_file models/pretrain_self_100_local --output_model_file models/pretrain_self_100_sup_200_local  --device 0 --local > logs/pretrain_self_100_sup_200_local.txt
```
## BootstrapGAN
### with multi-view pretraining:
```bash
python -u bootstrap.py --dataset dataset/CoNLL --n_iter 20 --min_match 2 --device 0 --local > logs/conll_local.txt
```
or 
```bash
python -u bootstrap.py --dataset dataset/OntoNotes --n_iter 20 --device 0 --local > logs/onto_local.txt
```
### with self-pretraining:
```bash
python -u bootstrap.py --input_model_file models/ul_weight1e-1/pretrain_self_100_sup_200_local --dataset dataset/CoNLL --min_match 2 --n_iter 20 --device 0 --local > logs/conll_100_200_local.txt
```
or 
```bash
python -u bootstrap.py --input_model_file models/ul_weight1e-2/pretrain_self_100_sup_200_local --dataset dataset/OntoNotes --n_iter 20 --device 0 --local > logs/onto_100_200_local.txt
```
