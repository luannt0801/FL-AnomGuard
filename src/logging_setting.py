import logging
import os
import yaml
# from src.add_config import *
from datetime import datetime

# log_directory = "logs"
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Lấy thời gian hiện tại
# log_file_path = os.path.join(log_directory, f"FedSCP-{timestamp}.log")
# os.makedirs(log_directory, exist_ok=True)

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    green = '\033[1;32;48m'
    cyan = '\033[1;36;48m'
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger("FedCSP-2024")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
)
# fh = logging.FileHandler(log_file_path)
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(file_formatter)
# logger.addHandler(fh)

# Test log messages
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")