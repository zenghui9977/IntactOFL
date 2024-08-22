import yaml
import logging
import os
import time
import sys
import torch
import numpy as np
import random
import copy
from visdom import Visdom
import csv
from torch.optim import Optimizer
import torchvision
from tqdm import tqdm, trange
import math
import abc


def init_logger(log_level):
    """Initialize internal logger of EasyFL.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG
    """
    log_formatter = logging.Formatter("%(asctime)-19.19s [%(levelname)s]  %(message)s")
    root_logger = logging.getLogger()

    log_level = logging.INFO if not log_level else log_level
    root_logger.setLevel(log_level)

    file_path = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, time.strftime("%Y%m%d_%H%M%S") + ".log")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    return logger

logger = init_logger(logging.INFO)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            logger.info(f"Load config from {file_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(e)
    

def save_yaml_config(file_path, config):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)
        logger.info(f"Save config to {file_path}")