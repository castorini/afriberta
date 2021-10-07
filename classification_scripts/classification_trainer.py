import json
import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Any
from typing import Dict
from typing import Optional

from classification_dataset import ClassificationDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import Trainer
from transformers import TrainingArguments
from transformers import XLMRobertaTokenizer

from src.utils import create_logger

MODEL_MAX_LENGTH = 512
SPLITS_MAP = {"train": "train_clean.tsv", "dev": "dev.tsv", "test": "test.tsv"}

NEWS_LABELS = {
    "yoruba": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics"],
    "hausa": ["africa", "world", "health", "nigeria", "politics"],
}
KEYS_NOT_IN_TRAIN_ARGS = [
    "model_dir",
    "tok_dir",
    "data_dir",
    "language",
    "max_seq_length",
]


class ClassificationTrainer:
    def __init__(self, args: Namespace) -> None:
        self.params = args
        if os.path.isdir(self.params.output_dir):
            raise ValueError(
                f"Output directory - {self.params.output_dir} - already exists, please delete or specify a new one "
            )
        os.makedirs(self.params.output_dir)

        tokenizer_class = AutoTokenizer
        if not self.params.tok_dir == "bert-base-multilingual-cased":
            # AfriBERTa's trained spm tokenizer model does not work with autotokenizer
            # out of the box, so we have to use the model-specific tokenizer
            tokenizer_class = XLMRobertaTokenizer

        self.tokenizer = tokenizer_class.from_pretrained(self.params.tok_dir)
        self.tokenizer.model_max_length = self.params.max_seq_length
        self.logger = create_logger(os.path.join(self.params.output_dir, "train_log.txt"))

        self._create_data()

    def _create_data(self) -> None:

        self.logger.info("Creating datasets...")

        self.labels = NEWS_LABELS[self.params.language]
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.train_dataset = ClassificationDataset(
            os.path.join(self.params.data_dir, SPLITS_MAP["train"]), self.tokenizer, self.label2id
        )
        self.eval_dataset = ClassificationDataset(
            os.path.join(self.params.data_dir, SPLITS_MAP["dev"]), self.tokenizer, self.label2id
        )
        self.test_dataset = ClassificationDataset(
            os.path.join(self.params.data_dir, SPLITS_MAP["test"]), self.tokenizer, self.label2id
        )

    def create_model(self) -> None:

        self.logger.info("Building model...")

        config = AutoConfig.from_pretrained(
            self.params.model_dir,
            num_labels=len(self.labels),
            id2label={str(i): label for i, label in enumerate(self.labels)},
            label2id=self.label2id,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.params.model_dir, config=config
        )

    @staticmethod
    def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self) -> None:
        self.create_model()
        self.initialize_train_args()
        self.logger.info(
            f"Starting Training with the following arguments:\n {json.dumps(vars(self.training_args), indent=2, default=str)}"
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        self.trainer.train()
        self.logger.info("Saving model...")
        self.trainer.save_model()
        self.trainer.state.save_to_json(os.path.join(self.params.output_dir, "trainer_state.json"))
        self.logger.info("Evaluating...")
        self.evaluate()

    def _evaluate(self, mode: str, dataset: Optional[Dataset] = None) -> None:
        """
        Perform evaluation on a given dataset.
        """
        if mode == "test":
            dataset = self.test_dataset
        eval_output = self.trainer.evaluate(dataset, metric_key_prefix=mode)
        output_eval_file = os.path.join(self.params.output_dir, f"{mode}_results.txt")
        with open(output_eval_file, "w") as writer:
            for key, value in sorted(eval_output.items()):
                writer.write(f"{key} = {value}\n")

    def evaluate(self) -> None:
        self._evaluate(mode="eval")
        self._evaluate(mode="test")

    def initialize_train_args(self) -> TrainingArguments:
        self.training_args = TrainingArguments(**self._get_train_args_kwargs())

    def _get_train_args_kwargs(self) -> Dict[str, Any]:
        kwargs = vars(self.params)
        for key in KEYS_NOT_IN_TRAIN_ARGS:
            if key in kwargs:
                del kwargs[key]
        return kwargs


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--tok_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=50000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.00005)

    args = parser.parse_args()

    trainer = ClassificationTrainer(args)
    trainer.train()
