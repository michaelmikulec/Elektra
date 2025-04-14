import math
import torch
import torch.nn as nn

class EEGTransformer(nn.Module):
  def __init__(
    self,
    input_dim   = 20,
    model_dim   = 256,
    num_heads   = 8,
    num_layers  = 6,
    ff_dim      = 1024,
    dropout     = 0.1,
    num_classes = 6,
    max_len     = 2000
  ):
    super().__init__()
    self.projector = nn.Linear(input_dim, model_dim)
    pe             = torch.zeros(max_len, model_dim)
    position       = torch.arange(0, max_len).unsqueeze(1)
    div_term       = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
    pe[:, 0::2]    = torch.sin(position * div_term)
    pe[:, 1::2]    = torch.cos(position * div_term)
    self.register_buffer("pe", pe.unsqueeze(0))

    encoder_layer = nn.TransformerEncoderLayer(
      d_model         = model_dim,
      nhead           = num_heads,
      dim_feedforward = ff_dim,
      dropout         = dropout,
      batch_first     = True
    )
    self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.classifier = nn.Sequential(
      nn.LayerNorm(model_dim),
      nn.Linear(model_dim, 128),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(128, num_classes)
    )

  def forward(self, x):
    x = self.projector(x)
    x = x + self.pe[:, :x.size(1)]
    x = self.encoder(x)
    x = x.mean(dim=1)
    return self.classifier(x)
