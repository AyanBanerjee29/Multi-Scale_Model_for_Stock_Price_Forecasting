import torch
import torch.nn as nn
from .msgnet import MSGNet
from .transformer import TransformerEncoder, TransformerDecoder


class MSGformer(nn.Module):
    """MSGformer: Hybrid Multi-Scale Graph-Transformer Architecture"""

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.d_model = config['model']['d_model']
        self.input_dim = config['model']['input_dim']
        self.pred_len = config['training']['pred_len']

        # Input embedding
        self.input_projection = nn.Linear(self.input_dim, self.d_model)

        # MSGNet for multi-scale feature extraction
        self.msgnet = MSGNet(config)

        # Transformer encoder for global dependencies
        self.encoder = TransformerEncoder(config)

        # Transformer decoder
        self.decoder = TransformerDecoder(config)

        # Output projection
        self.output_projection = nn.Linear(self.d_model, 1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Source tensor [B, L_src, input_dim]
            tgt: Target tensor [B, L_tgt, input_dim] (optional for training)
        Returns:
            Predictions [B, pred_len, 1]
        """
        B, L_src, _ = src.shape

        # Input embedding
        src_embedded = self.input_projection(src)  # [B, L_src, d_model]

        # Extract multi-scale features with MSGNet
        src_msgnet = self.msgnet(src_embedded)  # [B, L_src, d_model]

        # Encode with Transformer
        memory = self.encoder(src_msgnet)  # [B, L_src, d_model]

        # Prepare decoder input
        if tgt is None:
            # Inference mode: use last encoder output as initial decoder input
            tgt_input = memory[:, -1:, :].repeat(1, self.pred_len, 1)
        else:
            # Training mode: use target
            tgt_input = self.input_projection(tgt)

        # Decode
        output = self.decoder(tgt_input, memory)  # [B, pred_len, d_model]

        # Project to output
        predictions = self.output_projection(output)  # [B, pred_len, 1]

        return predictions

