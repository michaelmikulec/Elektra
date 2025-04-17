import torch
from dl.tf import tf

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_properties(0))

tf.train(
  index_csv       = "data/pidx.csv",
  data_dir        = "data/prep",
  batch_size      = 60,
  num_samples     = 6000,
  num_epochs      = 10,
  lr              = 1e-4,
  val_split       = 0.2,
  save_dir        = "dl/tf/models",
  log_file        = "dl/tf/logs/training.log",
  metrics_out     = "dl/tf/logs/training_metrics.parquet",
  resume_training = True
)