import os
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import create_logger

MIN_NUM_TOKENS = 5
NUM_GPUS = 1
INIT_DATA_SEED = 999
afriberta_tokenizer = AutoTokenizer.from_pretrained("castorini/afriberta_large")


class TrainDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        train_data_file: str,
        batch_size: int,
        experiment_path: str,
    ):
        super(TrainDataset, self).__init__()
        self.logger = create_logger(
            os.path.join(experiment_path, "data_log.txt"), "data_log"
        )
        self.logger.propagate = False

        self.batch_size = batch_size * NUM_GPUS
        self.data_seed = INIT_DATA_SEED
        self.num_examples_per_language = OrderedDict()
        self.train_data_file = train_data_file

        assert os.path.exists(self.train_data_file) == True, "Train file does not exist"

        lines = []
        with open(self.train_data_file, "r") as corpus:
            for idx, line in enumerate(tqdm(corpus.readlines())):
                info = json.loads(line)
                docid = info["id"]
                text = info["contents"]
                lines.append(text.lower())

        encoding = tokenizer(
            lines,
            max_length=512,
            add_special_tokens=True,
            truncation=True,
        )

        self.inputs = np.array(
            [
                {"input_ids": torch.tensor(ids, dtype=torch.long)}
                for ids in encoding["input_ids"]
            ]
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]


if __name__ == "__main__":
    train_data_file = (
        "/store/scratch/odunayo/mrtydi/mrtydi-v1.0-swahili/collection/docs.jsonl"
    )
    batch_size = 32
    experiment_path = "/store/scratch/odunayo/model-finetuning/model_checkpoints"
    dataset = TrainDataset(
        tokenizer=afriberta_tokenizer,
        train_data_file=train_data_file,
        batch_size=batch_size,
        experiment_path=experiment_path,
    )
    print(len(dataset))
    print(dataset[0].shape)
