import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class ScaleIdentification(nn.Module):
    """FFT-based scale identification module"""

    def __init__(self, k_scales: int):
        super().__init__()
        self.k_scales = k_scales

    def forward(self, x: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """
        Args:
            x: Input tensor [B, L, D]
        Returns:
            scales: List of k dominant scales
            amplitudes: Amplitude tensor [B, L//2]
        """
        B, L, D = x.shape

        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=1)
        amplitudes = torch.abs(x_fft).mean(dim=-1)  # [B, L//2+1]

        # Find top-k frequencies
        _, top_k_indices = torch.topk(amplitudes.mean(dim=0), self.k_scales)
        frequencies = top_k_indices.cpu().numpy()

        # Convert frequencies to scales
        scales = [max(1, L // (f + 1)) for f in frequencies]

        return scales, amplitudes


class AdaptiveGraphLayer(nn.Module):
    """Adaptive graph convolution layer"""

    def __init__(self, d_model: int, adj_powers: List[int] = [1, 2, 3]):
        super().__init__()
        self.d_model = d_model
        self.adj_powers = adj_powers

        # Learnable parameters for adjacency matrix
        self.E1 = nn.Parameter(torch.randn(d_model, d_model))
        self.E2 = nn.Parameter(torch.randn(d_model, d_model))

        # Output projection
        self.proj = nn.Linear(d_model * len(adj_powers), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, S, D] where N is number of variables
        Returns:
            Output tensor [B, N, S, D]
        """
        B, N, S, D = x.shape

        # Generate adaptive adjacency matrix
        adj = F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)  # [D, D]

        # Reshape for graph convolution
        x_flat = x.reshape(B * N * S, D)  # [B*N*S, D]

        # Apply multiple hops of graph convolution
        outputs = []
        adj_power = torch.eye(D, device=x.device)

        for p in self.adj_powers:
            for _ in range(p):
                adj_power = adj_power @ adj
            out = x_flat @ adj_power  # [B*N*S, D]
            outputs.append(out)

        # Concatenate and project
        out = torch.cat(outputs, dim=-1)  # [B*N*S, D*len(adj_powers)]
        out = self.proj(out)  # [B*N*S, D]

        # Reshape back
        out = out.reshape(B, N, S, D)

        return out


class ScaleGraphBlock(nn.Module):
    """Single ScaleGraph block"""

    def __init__(self, d_model: int, n_heads: int, dropout: float, adj_powers: List[int]):
        super().__init__()

        self.graph_layer = AdaptiveGraphLayer(d_model, adj_powers)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, S, D]
        Returns:
            Output tensor [B, N, S, D]
        """
        B, N, S, D = x.shape

        # Graph convolution for inter-series correlation
        x_graph = self.graph_layer(x)
        x = self.norm1(x + self.dropout(x_graph))

        # Multi-head attention for intra-series correlation
        x_flat = x.reshape(B * N, S, D)
        x_attn, _ = self.mha(x_flat, x_flat, x_flat)
        x_attn = x_attn.reshape(B, N, S, D)
        x = self.norm2(x + self.dropout(x_attn))

        return x


class MSGNet(nn.Module):
    """Multi-Scale Graph Neural Network"""

    def __init__(self, config: dict):
        super().__init__()

        self.d_model = config['model']['d_model']
        self.k_scales = config['model']['k_scales']
        self.n_blocks = config['msgnet']['n_blocks']

        self.scale_id = ScaleIdentification(self.k_scales)

        # Create ScaleGraph blocks
        self.blocks = nn.ModuleList([
            ScaleGraphBlock(
                self.d_model,
                config['model']['n_heads'],
                config['model']['dropout'],
                config['msgnet']['adj_powers']
            )
            for _ in range(self.n_blocks)
        ])

        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(self.k_scales)
        ])

    def reshape_to_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Reshape input to specific scale"""
        B, L, D = x.shape

        # Pad if necessary
        pad_len = (scale - L % scale) % scale
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            L = L + pad_len

        # Reshape to [B, N=1, S=scale, F=L//scale, D]
        n_frames = L // scale
        x_reshaped = x.reshape(B, 1, scale, n_frames, D)
        x_reshaped = x_reshaped.transpose(2, 3).reshape(B, 1, n_frames, scale * D)

        return x_reshaped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
        Returns:
            Output tensor [B, L, D]
        """
        B, L, D = x.shape

        # Identify dominant scales
        scales, amplitudes = self.scale_id(x)

        # Process each scale
        scale_outputs = []
        scale_weights = []

        for i, scale in enumerate(scales):
            # Reshape to scale
            x_scale = self.reshape_to_scale(x, scale)

            # Apply ScaleGraph blocks
            for block in self.blocks:
                x_scale = block(x_scale)

            # Project and reshape back
            x_scale = x_scale.reshape(B, -1, D)
            x_scale = self.scale_projections[i](x_scale)

            # Interpolate to original length
            if x_scale.size(1) != L:
                x_scale = F.interpolate(
                    x_scale.transpose(1, 2),
                    size=L,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            scale_outputs.append(x_scale)

            # Get weight from amplitude
            freq_idx = L // scale
            if freq_idx < amplitudes.size(1):
                weight = amplitudes[:, freq_idx].mean()
            else:
                weight = torch.tensor(0.1, device=x.device)
            scale_weights.append(weight)

        # Weighted aggregation
        scale_weights = torch.softmax(torch.stack(scale_weights), dim=0)
        output = sum(w * out for w, out in zip(scale_weights, scale_outputs))

        return output

