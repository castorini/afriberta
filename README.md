# AfriBERTa: Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages

This repository contains the code for the paper [***Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages***](https://aclanthology.org/2021.mrl-1.11/) which appears in the first workshop on Multilingual Representation Learning at EMNLP 2021. 

AfriBERTa was trained on 11 languages - Afaan  Oromoo (also  called  Oromo), Amharic, Gahuza (a mixed language containing Kinyarwanda and Kirundi), Hausa, Igbo, Nigerian Pidgin, Somali, Swahili, Tigrinya and Yorùbá.
AfriBERTa was evaluated on NER and text classification spanning 10 languages (some of which it was not pretrained on).
It outperformed mBERT and XLM-R on several languages and is very competitive overall.


## Pretrained Models and Datasets

**Models:**

We release the following pretrained models:

- [AfriBERTa Small](https://huggingface.co/castorini/afriberta_small) (97M params)
- [AfriBERTa Base](https://huggingface.co/castorini/afriberta_base) (111M params)
- [AfriBERTa Large](https://huggingface.co/castorini/afriberta_large) (126M params)

**Dataset**:

https://huggingface.co/datasets/castorini/afriberta

## Reproducing Experiments

### Datasets and Tokenizer
Below are details on how to obtain the datasets and trained sentencepiece tokenizer:

**Language Modelling**: The data for language modelling can be downloaded from [this URL](https://huggingface.co/datasets/castorini/afriberta)

**NER**: To obtain the NER dataset, please download it from [this repository](https://github.com/masakhane-io/masakhane-ner)

**Text Classification**: To obtain the topic classification dataset, please download it from [this repository](https://github.com/uds-lsv/transfer-distant-transformer-african)

**Tokenizer**: The trained sentencepiece tokenizer can be downloaded from [this URL](https://drive.google.com/file/d/1-wwAGgGG9iMFfj-85lVWq0sj-iEaxD-g/view?usp=sharing)


### Training

To train AfriBERTa and evaluate on both downstream tasks, simply install all requirements in ```requirements.txt```, download the relevant datasets and run the following script:

```
bash run_all.sh
```

This script will: 
1. Train the multilingual language model from scratch and save the model as well as relevant logs
2. Evaluate the trained language model on NER for all ten languages over 5 seeds
3. Evaluate the trained language model on text classification for all two languages over 5 seeds


## Citation
```
@inproceedings{ogueji-etal-2021-small,
    title = "Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages",
    author = "Ogueji, Kelechi  and
      Zhu, Yuxin  and
      Lin, Jimmy",
    booktitle = "Proceedings of the 1st Workshop on Multilingual Representation Learning",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.mrl-1.11",
    pages = "116--126",
}
```
