import logging
from logging import Logger
from typing import Any
from typing import Dict
from typing import Optional

import yaml


def create_logger(log_file: str, name: Optional[str] = None) -> Logger:
    """
    Create logger for logging the experiment process.
    """
    if not name:
        name = __name__

    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)

    return logger


def load_config(filename: str) -> Dict[str, Any]:
    """
    loads yaml configuration file.
    """
    conf_file = yaml.full_load(open(filename, "r"))
    return conf_file
