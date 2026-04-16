from typing import Optional
import torch
import torch.nn as nn
from torch.nn import (
    MultiheadAttention,
    Linear,
    Dropout,
    LayerNorm,
    ReLU,
    Parameter,
    TransformerEncoderLayer,
)
from lightning.pytorch import LightningModule
from src.utils.model_utils import embedding_to_padding


class PolygenEncoderLayer(TransformerEncoderLayer):
    """Encoder module as described in the Polygen paper"""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        re_zero: bool = True,
    ) -> None:
        """Initializes PolygenEncoderLayer

        Args:
            d_model: Size of the embedding vectors.
            nhead: Number of multihead attention heads.
            dim_feedforward: size of fully connected layer.
            dropout: Dropout rate applied after ReLU in each connected layer.
            re_zero: If True, Alpha scale residuals with zero initialization.
        """
        super(PolygenEncoderLayer, self).__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.dropout = Dropout(dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.activation = ReLU()

        self.re_zero = re_zero
        self.alpha = Parameter(data=torch.Tensor([0.0]))
        self.beta = Parameter(data=torch.Tensor([0.0]))

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = False,
    ) -> torch.Tensor:
        """Forward method for the PolygenEncoderLayer

        Args:
            src: A Tensor of shape [sequence_length, batch_size, embed_size]. Input Tensor to the TransformerEncoder
            src_mask: A Tensor of shape [sequence_length, sequence_length]. The mask for the input sequence
            src_key_padding_mask: A Tensor of shape [sequence_length, batch_size]. Tells attention which
                                  aspects of the input sequence to ignore due to them being padding

        Returns:
            src: A Tensor of shape [sequence_length, batch_size, embed_size]
        """
        # if is_causal and src_mask is None:
        #     # 如果is_causal为True且没有提供src_mask，则生成一个方形后续掩码
        #     src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)

        src_n1 = self.norm1(src)
        src2 = self.self_attn(
            src_n1,
            src_n1,
            src_n1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal
        )[0]
        if self.re_zero:
            src2 = src2 * self.alpha
        src2 = self.dropout(src2)
        src = src + src2

        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.linear2(src2)
        if self.re_zero:
            src2 = src2 * self.beta
        src2 = self.dropout(src2)
        src = src + src2
        return src


class PolygenEncoder(LightningModule):
    """A modified version of the traditional Transformer Encoder suited for Polygen input sequences"""

    def __init__(
        self,
        hidden_size: int = 256,
        fc_size: int = 1024,
        num_heads: int = 4,
        num_layers: int = 8,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initializes the PolygenEncoder

        Args:
            hidden_size: Size of the embedding vectors.
            fc_size: Size of the fully connected layer.
            num_heads: Number of multihead attention heads.
            layer_norm: Boolean variable that signifies if layer normalization should be used.
            num_layers: Number of decoder layers in the decoder.
            dropout_rate: Dropout rate applied immediately after the ReLU in each fully connected layer.
        """
        super(PolygenEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.TransformerEncoder(
            PolygenEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=fc_size,
                dropout=dropout_rate,
            ),
            num_layers=num_layers,
        )
        self.norm = LayerNorm(hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward method for the Transformer Encoder

        Args:
            inputs: A Tensor of shape [sequence_length, batch_size, embed_size]. Represents the input sequence.

        Returns:
            outputs: A Tensor of shape [sequence_length, batch_size, embed_size]. Represents the result of the TransformerEncoder
        """
        padding_mask = embedding_to_padding(inputs)
        out = self.encoder(inputs, src_key_padding_mask=padding_mask, is_causal=False)
        return self.norm(out)
