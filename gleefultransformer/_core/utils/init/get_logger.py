import logging
from logging import Logger


def get_logger() -> Logger:
    logger = logging.getLogger("gleefultransformer")
    handler = logging.StreamHandler()
    handler.addFilter(logging.Filter("gleefultransformer"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[handler],
    )
    return logger
