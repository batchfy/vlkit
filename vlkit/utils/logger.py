import logging
import os
import os.path as osp


def get_logger(name: str, log_file: str, log_level=logging.INFO):
    os.makedirs(osp.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, 'w')
    handlers = [stream_handler, file_handler]
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger
