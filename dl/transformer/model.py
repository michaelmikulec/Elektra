import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
  
  def __init__(self, d_model, max_len=10000):
    super().__init__()
    pos_data = torch.zeros(max_len, d_model)
    pos_range = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pos_data[:, 0::2] = torch.sin(pos_range * div_term)
    pos_data[:, 1::2] = torch.cos(pos_range * div_term)
    pos_data = pos_data.unsqueeze(0)
    self.register_buffer("pos_data", pos_data)
  
  def forward(self, x):
    seq_len = x.size(1)
    if seq_len > self.pos_data.size(1):
      raise ValueError(str(seq_len))
    x = x + self.pos_data[:, :seq_len, :]
    return x

class LearnablePositionalEncoding(nn.Module):
  
  def __init__(self, d_model, max_len=10000):
    super().__init__()
    self.pos_data = nn.Parameter(torch.zeros(1, max_len, d_model))
  
  def forward(self, x):
    seq_len = x.size(1)
    if seq_len > self.pos_data.size(1):
      raise ValueError(str(seq_len))
    x = x + self.pos_data[:, :seq_len, :]
    return x

class EEGTransformer(nn.Module):
  
  def __init__(
    self,
    input_dim = 20,
    model_dim = 128,
    num_heads = 4,
    num_layers = 2,
    dim_feedforward = 256,
    dropout = 0.1,
    num_classes = 6,
    max_len = 10000,
    use_learnable_pos_emb = True,
    use_cls_token = True,
    pooling = "cls"
  ):
    super().__init__()
    self.use_cls_token = use_cls_token
    self.pooling = pooling
    if use_cls_token:
      self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
    self.input_embed = nn.Linear(input_dim, model_dim)
    if use_learnable_pos_emb:
      self.pos_encoding = LearnablePositionalEncoding(model_dim, max_len=max_len)
    else:
      self.pos_encoding = SinusoidalPositionalEncoding(model_dim, max_len=max_len)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=model_dim,
      nhead=num_heads,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      activation="relu",
      batch_first=True
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.classifier = nn.Sequential(
      nn.LayerNorm(model_dim),
      nn.Linear(model_dim, model_dim // 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(model_dim // 2, num_classes)
    )
    self.init_weights()
  
  def init_weights(self):
    if self.use_cls_token:
      nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
  
  def forward(self, x):
    x = self.input_embed(x)
    if self.use_cls_token:
      batch_size = x.size(0)
      cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
      x = torch.cat([cls_tokens, x], dim=1)
    x = self.pos_encoding(x)
    x = self.encoder(x)
    if self.use_cls_token and self.pooling == "cls":
      x = x[:, 0, :]
    else:
      x = x.mean(dim=1)
    x = self.classifier(x)
    return x

