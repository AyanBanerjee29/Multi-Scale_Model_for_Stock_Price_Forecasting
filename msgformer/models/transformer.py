import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, L, D]
        Returns:
            Tensor [B, L, D]
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""

    def __init__(self, config: dict):
        super().__init__()

        d_model = config['model']['d_model']
        n_heads = config['model']['n_heads']
        n_layers = config['model']['n_encoder_layers']
        dropout = config['model']['dropout']

        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            mask: Attention mask
        Returns:
            Output tensor [B, L, D]
        """
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder"""

    def __init__(self, config: dict):
        super().__init__()

        d_model = config['model']['d_model']
        n_heads = config['model']['n_heads']
        n_layers = config['model']['n_decoder_layers']
        dropout = config['model']['dropout']

        self.pos_encoding = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tgt: Target tensor [B, L_tgt, D]
            memory: Memory tensor from encoder [B, L_src, D]
            tgt_mask: Target mask
            memory_mask: Memory mask
        Returns:
            Output tensor [B, L_tgt, D]
        """
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output

