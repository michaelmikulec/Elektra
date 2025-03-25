import math,torch,torch.nn as nn,torch.nn.functional as F

class PositionalEncoding(nn.Module):
  def __init__(self,d_model,max_len=12000):
    super(PositionalEncoding,self).__init__();
    pe=torch.zeros(max_len,d_model);
    position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1);
    div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model));
    pe[:,0::2]=torch.sin(position*div_term);
    pe[:,1::2]=torch.cos(position*div_term);
    pe=pe.unsqueeze(0);
    self.register_buffer("pe",pe);

  def forward(self,x):
    seq_len=x.size(1);
    if seq_len>self.pe.size(1):raise ValueError(f"Input sequence length {seq_len} exceeds maximum length {self.pe.size(1)} in PositionalEncoding. Increase max_len or downsample the data.");
    x=x + self.pe[:,:seq_len,:]; 
    return x;

class EEGTransformer(nn.Module):
  def __init__(self,input_dim=20,model_dim=128,num_heads=4,num_layers=2,dim_feedforward=256,dropout=0.1,num_classes=6):
    super(EEGTransformer,self).__init__();
    self.input_embed=nn.Linear(input_dim,model_dim);
    self.pos_encoding=PositionalEncoding(d_model=model_dim,max_len=12000);
    encoder_layer=nn.TransformerEncoderLayer(d_model=model_dim,nhead=num_heads,dim_feedforward=dim_feedforward,dropout=dropout,activation="relu",batch_first=True);
    self.transformer_encoder=nn.TransformerEncoder(encoder_layer,num_layers=num_layers);
    self.fc_out=nn.Linear(model_dim,num_classes);

  def forward(self,x):
    x=self.input_embed(x);
    x=self.pos_encoding(x); 
    x=self.transformer_encoder(x);
    x=x.mean(dim=1);
    out=self.fc_out(x);
    return out;