import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=12000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum length {self.pe.size(1)}. "
                "Increase max_len or downsample the data."
            )
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=12000):
        super().__init__()
        # Initialize learnable parameters of shape [1, max_len, d_model]
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum length {self.pe.size(1)}. "
                "Increase max_len or downsample the data."
            )
        # Add learnable positional encoding
        x = x + self.pe[:, :seq_len, :]
        return x

class EEGTransformer(nn.Module):
    def __init__(
        self,
        input_dim=20,
        model_dim=128,
        num_heads=4,  
        num_layers=2, 
        dim_feedforward=256,  
        dropout=0.1,
        num_classes=6,
        max_len=12000,
        use_learnable_pos_emb=True,
        use_cls_token=True,
        pooling="cls"
    ):
        """
        Args:
            input_dim:       Number of input features (e.g., EEG channels).
            model_dim:       Dimension of the transformer model.
            num_heads:       Number of attention heads.
            num_layers:      Number of stacked TransformerEncoder layers.
            dim_feedforward: Hidden dimension of the feedforward network inside the Transformer.
            dropout:         Dropout probability.
            num_classes:     Number of output classes.
            max_len:         Maximum sequence length for positional encoding.
            use_learnable_pos_emb: If True, use learnable embeddings; otherwise use sinusoidal.
            use_cls_token:   If True, prepend a learnable [CLS] token to each sequence.
            pooling:         If "cls", output is taken from the [CLS] token; otherwise, mean-pool.
        """
        super().__init__()

        self.use_cls_token = use_cls_token
        self.pooling = pooling

        # Learnable [CLS] token (optional)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))

        # Input embedding: project input_dim -> model_dim
        self.input_embed = nn.Linear(input_dim, model_dim)

        # Positional encoding (either learnable or fixed sinusoidal)
        if use_learnable_pos_emb:
            self.pos_encoding = LearnablePositionalEncoding(model_dim, max_len=max_len)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(model_dim, max_len=max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True  # So the Transformer expects shape [batch_size, seq_len, model_dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head: small MLP
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),           # Stabilize training
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

        # Initialize parameters (especially the CLS token) with a normal or Xavier initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02) if self.use_cls_token else None

    def forward(self, x):
        # 1. Project input to model_dim
        x = self.input_embed(x)  # [batch_size, seq_len, model_dim]

        # 2. Optionally prepend [CLS] token
        if self.use_cls_token:
            bsz = x.size(0)
            cls_tokens = self.cls_token.repeat(bsz, 1, 1)  # [batch_size, 1, model_dim]
            x = torch.cat([cls_tokens, x], dim=1)          # [batch_size, 1 + seq_len, model_dim]

        # 3. Add positional encoding
        x = self.pos_encoding(x)  # [batch_size, seq_len (+1), model_dim]

        # 4. Pass through Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, seq_len (+1), model_dim]

        # 5. Pooling strategy:
        if self.use_cls_token and self.pooling == "cls":
            # Take the [CLS] token output
            x = x[:, 0, :]  # [batch_size, model_dim]
        else:
            # Mean-pool over the time dimension
            x = x.mean(dim=1)  # [batch_size, model_dim]

        # 6. Classification head
        logits = self.classifier(x)  # [batch_size, num_classes]
        return logits

if __name__ == "__main__":
    # Fake data: batch_size=8, seq_len=1000, input_dim=20
    fake_data = torch.randn(8, 1000, 20)

    model = EEGTransformer(
        input_dim=20,
        model_dim=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=6,
        max_len=12000,
        use_learnable_pos_emb=True,
        use_cls_token=True,
        pooling="cls"
    )

    outputs = model(fake_data)
    print(outputs.shape)  # Should be [8, 6]
