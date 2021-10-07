import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer

from src.utils import create_logger

TRAIN_FILE_PATTERN = "train.*"
INIT_DATA_SEED = 999
MIN_NUM_TOKENS = 5
NUM_GPUS = 2
np.random.seed(INIT_DATA_SEED)


class TrainDataset(Dataset):
    """
    Custom dataset that contains all languages and implements random sampling of a language to train
    on per batch. Following XLM-R, we train on one language per batch.

    Note that, because of the small size of our data, we decide to hold all of it in memory.
    """

    def __init__(
        self,
        tokenizer: XLMRobertaTokenizer,
        train_data_dir: str,
        batch_size: int,
        experiment_path: str,
        lang_sampling_factor: Union[int, float] = 1.0,
    ) -> None:

        self.logger = create_logger(os.path.join(experiment_path, "data_log.txt"), "data_log")
        self.logger.propagate = False

        self.lang_sampling_factor = lang_sampling_factor
        self.batch_size = batch_size * NUM_GPUS
        self.data_seed = INIT_DATA_SEED
        self.sampling_counter = 0
        self.examples = {}
        self.languages: List[str] = []
        self.num_examples_per_language = OrderedDict()

        file_paths = Path(train_data_dir).glob(TRAIN_FILE_PATTERN)
        for file_path in file_paths:
            language = file_path.suffix.replace(".", "")
            lines = [
                line
                for line in file_path.read_text(encoding="utf-8").splitlines()
                if (len(line.split()) > MIN_NUM_TOKENS and not line.isspace())
            ]

            encoding = tokenizer(
                lines,
                max_length=tokenizer.model_max_length,
                add_special_tokens=True,
                truncation=True,
            )

            inputs = np.array(
                [
                    {"input_ids": torch.tensor(ids, dtype=torch.long)}
                    for ids in encoding["input_ids"]
                ]
            )

            self.num_examples_per_language[language] = len(inputs)
            self.examples[language] = inputs
            self.languages.append(language)

        self._set_language_probs()

    def create_language_index_mapping(self) -> None:
        """
        Create language to index mapping dictionary.
        """
        self.language_data_index_mapping = {}
        for language in self.languages:
            num_examples = len(self.examples[language])
            language_index_mapping = list(range(num_examples))
            np.random.shuffle(language_index_mapping)
            self.language_data_index_mapping[language] = language_index_mapping

    def __len__(self) -> int:
        """
        Total number of examples.
        """
        return sum(len(input_) for input_ in self.examples.values())

    def __getitem__(self, index: int) -> Dict[str, torch.tensor]:
        """
        Obtain one example of data.
        """
        if self.sampling_counter % self.batch_size == 0:
            self._sample_batch_language()
            current_batch_no = self.sampling_counter // self.batch_size
            self.logger.info(
                f"Worker {self.worker_id} : Language sampled for batch {current_batch_no} is {self.batch_language}"
            )
        self.sampling_counter += 1
        batch_index = self._get_random_index()
        return self.examples[self.batch_language][batch_index]

    def _sample_batch_language(self) -> None:
        """
        Sample a language to train on for a batch.
        """
        if not self.languages:  # check if all language examples have been exhausted
            self.logger.info(
                f"Worker {self.worker_id}: All language examples exhausted, recreating variables..."
            )
            self._recreate_language_sampling_variables()
        sampled_language_index = np.argmax(np.random.multinomial(1, self.language_probs))
        self.batch_language: str = self.languages[sampled_language_index]

    def _set_language_probs(self) -> None:
        """
        Initialize the sampling probabilities of languages based on the number of sentences for each
        language.

        We use this to control the order of batch languages seen by the model. Ideally, we want to
        maintain a diverse order of batch languages as much as possible The most diverse order is
        acheived by setting the factor to 1.0
        """
        if self.lang_sampling_factor <= 0:
            # if sampling factor is negative or set to 0, we sample following a uniform distribution
            self.language_probs = [
                1 / len(self.num_examples_per_language) for _ in self.num_examples_per_language
            ]
            return

        total_num_examples = len(self)
        probs = np.array(
            [
                (value / total_num_examples) ** self.lang_sampling_factor
                for value in self.num_examples_per_language.values()
            ]
        )
        self.language_probs = list(probs / probs.sum())
        self.logger.info(
            f"Language probs created as:\n {dict(zip(self.num_examples_per_language.keys(), self.language_probs))}"
        )

    def _get_random_index(self) -> int:
        """
        Return random data index from batch language index mapping.
        """
        try:
            return self.language_data_index_mapping[self.batch_language].pop()
        except IndexError:
            del self.language_probs[self.languages.index(self.batch_language)]
            del self.languages[self.languages.index(self.batch_language)]
            prev_batch_lang = self.batch_language
            self._sample_batch_language()
            msg = f"Worker {self.worker_id}: All data examples exhausted for language: {prev_batch_lang}. Newly sampled batch language set as: {self.batch_language}"
            self.logger.info(msg)
            return self._get_random_index()

    def set_worker_id(self, worker_id: int) -> None:
        """
        Set worker ID.
        """
        self.worker_id = worker_id

    def _recreate_language_sampling_variables(self) -> None:
        """
        Once all examples for all languages are exhausted, recreate needed language sampling
        variables.
        """
        self.languages = list(self.num_examples_per_language.keys())
        self._set_language_probs()
        self.create_language_index_mapping()


class EvalDataset(Dataset):
    """
    Simple line by line dataset for evaluation.
    """

    def __init__(self, tokenizer: XLMRobertaTokenizer, eval_file_path: str,) -> None:
        lines = [
            line
            for line in Path(eval_file_path).read_text(encoding="utf-8").splitlines()
            if (len(line.split()) > MIN_NUM_TOKENS and not line.isspace())
        ]
        encoding = tokenizer(
            lines, max_length=tokenizer.model_max_length, add_special_tokens=True, truncation=True,
        )
        self.examples = np.array(
            [{"input_ids": torch.tensor(ids, dtype=torch.long)} for ids in encoding["input_ids"]]
        )

    def __len__(self) -> int:
        """
        Total number of examples.
        """
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        """
        Obtain one example of data.
        """
        return self.examples[i]
