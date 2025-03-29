import os
import logging

def create_logger(logging_level=logging.DEBUG):
  logger = logging.getLogger();
  logger.propagate = False;
  logger.setLevel(logging_level);
  logFormatter = logging.Formatter(
    fmt=f"[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="%",
    validate=True
  );
  logHandler = logging.FileHandler(
    filename="./logs/logs.log",
    mode="a",
    encoding="utf-8",
    errors="strict",
    delay=False
  );
  logHandler.setFormatter(logFormatter);
  logger.addHandler(logHandler);
  return logger;

def build_workspace(logger=None) -> None:
  dirs = [
    "./logs/",
    "./data/",
    "./data/unprocessed_data/",
    "./data/unprocessed_data/eegs/",
    "./data/unprocessed_data/spectrograms/",
    "./data/processed_data/",
    "./data/processed_data/eegs/",
    "./data/processed_data/spectrograms/",
    "./data/training_data/",
    "./data/training_data/eegs/",
    "./data/training_data/spectrograms/",
    "./prep/",
    "./dl/",
    "./dl/transformer/",
    "./dl/cnn/",
    "./ml/",
    "./ui/"
  ];
  if logger: 
    logger.info("Building Workspace...")
  for dir in dirs:
    if not os.path.exists(dir):
      os.makedirs(dir);
      if logger: 
        logger.info(f"Created: {dir}");
    else:
      if logger: 
        logger.debug(f"Exists: {dir}");

if __name__ == "__main__":
  logger = create_logger(logging.DEBUG);
  build_workspace(logger);