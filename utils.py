import logging, os

logger = logging.getLogger();
logger.propagate = False;
logger.setLevel(logging.DEBUG);
logFormatter = logging.Formatter(
  fmt=f"[%(asctime)s] [%(levelname)s]: %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
  style="%",
  validate=True
);
logHandler = logging.FileHandler(
  filename=os.path.join(os.getcwd(), "logs", "logs.log"),
  mode="a",
  encoding="utf-8",
  errors="strict",
  delay=False
);
logHandler.setFormatter(logFormatter);
logger.addHandler(logHandler);