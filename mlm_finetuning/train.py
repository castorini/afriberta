import logging
import os
import argparse
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import transformers
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments

from transformers import AutoConfig
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments


from dataset import TrainDataset

parser = argparse.ArgumentParser(description="Finetuning Afriberta for Downstream")

parser.add_argument(
    "--tokenizer",
    default="castorini/afriberta_large",
    type=str,
    help="hugging face tokenizer",
)

parser.add_argument(
    "--model",
    default="castorini/afriberta_large",
    type=str,
    help="hugging face model or checkpoint folder",
)

parser.add_argument("--batch_size", default=32, type=int, help="training batch size")

parser.add_argument("--experiment_path", type=str, help="path to store checkpoint path")

parser.add_argument("--data_jsonl_path", type=str, help="path to jsonl data")

parser.add_argument(
    "--num_epochs", required=True, type=int, help="Number of training epochs"
)


class FinetuneTrainer:
    def __init__(
        self,
        tokenizer,
        experiment_path,
        batch_size,
        data_jsonl_path,
        num_epochs,
        model_name,
    ):
        self.data_jsonl_path = data_jsonl_path
        self.experiment_path = experiment_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.training_args = self._generate_training_args()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.dataset = TrainDataset(
            tokenizer=self.tokenizer,
            train_data_file=data_jsonl_path,
            batch_size=batch_size,
            experiment_path=experiment_path,
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=self.config
        )

    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.dataset,
        )
        trainer.train()
        self._save_model(trainer)

    def _generate_training_args(self):
        training_args = TrainingArguments(
            output_dir=self.experiment_path,
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            save_steps=200,
            prediction_loss_only=True,
            logging_strategy="steps",
            save_strategy="steps",
            report_to="tensorboard",
        )
        return training_args

    def _save_model(self, trainer):
        trainer.save_model(self.experiment_path)


if __name__ == "__main__":
    args = parser.parse_args()
    finetunetrainer = FinetuneTrainer(
        args.tokenizer,
        args.experiment_path,
        args.batch_size,
        args.data_jsonl_path,
        args.num_epochs,
        args.model,
    )
    finetunetrainer.train()
