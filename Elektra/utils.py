import logging
from config import config

logger = logging.getLogger();
logger.propagate = False;
logger.setLevel(config["logging_lvl"]);

logFormatter = logging.Formatter(
    fmt=f"[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="%",
    validate=True
);

logHandler = logging.FileHandler(
    filename=config['log_file'],
    mode="a",
    encoding="utf-8",
    errors="strict",
    delay=False
);

logHandler.setFormatter(logFormatter);
logger.addHandler(logHandler);