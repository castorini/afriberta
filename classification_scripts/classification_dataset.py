from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ClassificationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        label_map: Dict[str, int],
        sep: str = "\t",
    ) -> None:

        df = pd.read_csv(data_path, sep=sep)
        self.examples = []
        for sentence, label in zip(df.news_title, df.label):
            example = tokenizer(
                sentence,
                max_length=tokenizer.model_max_length,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
            )
            label = torch.tensor([label_map[label.lower()]])
            example["labels"] = label
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.examples[index]
