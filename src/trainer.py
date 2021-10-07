import logging
import math
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import transformers
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForWholeWordMask
from transformers import TrainingArguments
from transformers import XLMRobertaConfig
from transformers import XLMRobertaForMaskedLM
from transformers import XLMRobertaTokenizer

from src.custom import CustomTrainer
from src.dataset import EvalDataset
from src.dataset import TrainDataset
from src.utils import create_logger

DEFAULT_XLM_MODEL_SIZE = "xlm-roberta-base"
MLM_PROBABILITY = 0.15
EVAL_FILE_PATTERN = "eval.*"
KEYS_NOT_IN_TRAIN_ARGS = [
    "train_from_scratch",
    "use_whole_word_mask",
    "lang_sampling_factor",
    "resume_training",
]

transformers.logging.set_verbosity_debug()


class TrainingManager:
    """
    A class to manage the training and evaluation of the MLM.

    The actual training is done by a modified version (see custom.py) of the
    huggingface's trainer - https://huggingface.co/transformers/main_classes/trainer.html

    Args:
        config: Loaded configuration from specified yaml file
        experiment_path: path specified to save training outputs
    """

    def __init__(self, config: Dict[str, Any], experiment_path: str) -> None:
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.train_config = config["training"]
        self.train_config["output_dir"] = experiment_path
        self.logger = create_logger(os.path.join(experiment_path, "train_log.txt"))

        # modifying huggingface logger to log into a file
        hf_logger = transformers.logging.get_logger()
        file_handler = logging.FileHandler(os.path.join(experiment_path, "hf_log.txt"))
        file_handler.setLevel(level=logging.DEBUG)
        hf_logger.addHandler(file_handler)

        self.logger.info(f"Experiment Output Path: {experiment_path}")
        self.logger.info(f"Training will be done with this configuration: \n {config} ")

        self._maybe_resume_training()

    def _build_tokenizer(self) -> None:
        """
        Build tokenizer from pretrained sentencepiece model and update config.
        """
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            self.model_config.pop("tokenizer_path")
        )
        self.tokenizer.model_max_length = self.model_config["max_length"]

    def _build_model(self) -> None:
        """
        Build model from specified model config.
        """
        self.logger.info("Building model...")
        self._update_model_config()
        xlm_roberta_config = XLMRobertaConfig(**self.model_config)
        self.model = XLMRobertaForMaskedLM(xlm_roberta_config)
        self.logger.info(f"Model built with num parameters: {self.model.num_parameters()}")

    def _build_datasets(self) -> None:
        """
        Build dataset from supplied train and evaluation files.
        """
        self.logger.info("Building datasets...")
        batch_size = self.train_config["per_device_train_batch_size"]
        lang_sampling_factor = self.train_config.pop("lang_sampling_factor")
        self.logger.info(f"Building train dataset from {self.data_config['train']}...")
        self.train_dataset = TrainDataset(
            self.tokenizer,
            self.data_config["train"],
            batch_size,
            self.train_config["output_dir"],
            lang_sampling_factor=lang_sampling_factor,
        )
        self.logger.info(f"No. of training sentences: {len(self.train_dataset)}")
        self.logger.info(f"Building evaluation dataset from {self.data_config['eval']['all']}...")
        self.eval_dataset = EvalDataset(self.tokenizer, self.data_config["eval"]["all"],)
        self.logger.info(f"No. of evaluation sentences: {len(self.eval_dataset)}")

    def train(self) -> None:
        """
        Perform training.
        """
        self.logger.info("Starting Training...")

        data_collator = self.collator_class(
            tokenizer=self.tokenizer, mlm_probability=MLM_PROBABILITY
        )

        training_args = TrainingArguments(**self.train_config)
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        train_results = self.trainer.train(model_path=self.model_path)
        train_results_file = os.path.join(self.train_config["output_dir"], "train_results.txt")
        with open(train_results_file, "w") as writer:
            self.logger.info("***** Train results *****")
            for key, value in sorted(train_results.metrics.items()):
                self.logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        self.logger.info("Training Done! Saving model and model state...")
        self.trainer.save_model()
        self.trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )
        self.logger.info("Saving done!")
        self.evaluate()

    def evaluate(self) -> None:
        """
        Evaluate trained model on entire evaluation dataset and on per language datasets.
        """
        self.logger.info("Evaluating model...")
        self.logger.info("Evaluating on entire evaluation dataset...")
        self._evaluate()
        self.logger.info("Done! Evaluating on each language...")
        eval_dataset_path = Path(self.data_config["eval"]["per_lang"])
        eval_file_paths = eval_dataset_path.glob(EVAL_FILE_PATTERN)
        for file_path in eval_file_paths:
            language = file_path.suffix.replace(".", "")
            dataset = EvalDataset(self.tokenizer, str(file_path))
            self.logger.info(f"Evaluating {language} with {file_path}...")
            self._evaluate(dataset, language)
        self.logger.info("Completed all evaluations!")

    def _evaluate(self, eval_dataset: Optional[Dataset] = None, language: str = "all") -> None:
        """
        Perform evaluation on a given dataset.
        """
        eval_output = self.trainer.evaluate(eval_dataset)
        eval_output["perplexity"] = math.exp(eval_output["eval_loss"])

        output_eval_file = os.path.join(self.train_config["output_dir"], language + "_eval.txt")
        with open(output_eval_file, "w") as writer:
            self.logger.info(f"***** {language} eval results *****")
            for key, value in sorted(eval_output.items()):
                self.logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    def _maybe_resume_training(self) -> None:
        """
        Checks if we want to resume the training or not, and launches the appropriate option.
        """
        self._set_data_collator_class()

        if self.train_config.pop("resume_training", None):
            self.model_path = self.train_config["output_dir"]
            self.logger.info(f"Training will resume from {self.model_path}")
            self._build_tokenizer()
            self._build_datasets()
            self._remove_redundant_training_args()
            self.model = XLMRobertaForMaskedLM.from_pretrained(self.model_path)
            self.logger.info(
                f"Model loaded from {self.model_path} with num parameters: {self.model.num_parameters()}"
            )
        else:
            self.model_path = None
            if self.train_config.pop("train_from_scratch"):
                self.logger.info("Training from scratch...")
                self._build_tokenizer()
                self._build_model()
            else:
                self.logger.info("Not training from scratch, finetuning pretrained model...")
                self.logger.info("Building tokenizer from pretrained...")
                self.tokenizer = XLMRobertaTokenizer.from_pretrained(DEFAULT_XLM_MODEL_SIZE)
                self.logger.info("Building model from pretrained...")
                self.model = XLMRobertaForMaskedLM.from_pretrained(DEFAULT_XLM_MODEL_SIZE)
            self._build_datasets()

    def _remove_redundant_training_args(self) -> None:
        """
        Removes keys from self.train_config that are not accepted in huggingface's traininng
        arguments.
        """
        for key in KEYS_NOT_IN_TRAIN_ARGS:
            if key in self.train_config:
                del self.train_config[key]

    def _set_data_collator_class(self) -> None:
        """
        Set the data collator class.
        """
        if self.train_config.pop("use_whole_word_mask"):
            self.collator_class = DataCollatorForWholeWordMask
        else:
            self.collator_class = DataCollatorForLanguageModeling

    def _update_model_config(self) -> None:
        """
        Update model configuration.
        """
        self.model_config["vocab_size"] = self.tokenizer.vocab_size
        self.model_config["max_position_embeddings"] = self.model_config["max_length"] + 2
