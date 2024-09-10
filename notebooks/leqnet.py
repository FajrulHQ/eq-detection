import warnings

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import ActivationLSTMCell, CustomLSTM, WaveformModel
from blocks import *

# For implementation, potentially follow: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
class LEQNet(EQTransformer):
    def update_attributes(self, modified=False):
        self.modified = modified
        # customize encoder and decoder blocks
        self.encoder = EncoderSepConv1d(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples
        )

        decoder = DecoderSepConv1d(
            input_channels=16,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=self.in_samples,
        )
        self.decoder_d = decoder

        pick_decoders = [decoder for _ in range(self.classes)]
        self.pick_decoders = nn.ModuleList(pick_decoders)

        # customize res cnn blocks
        self.res_cnn_stack = DeeperBottleneckStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
        )

        # customize transformer blocks
        if self.modified:
            num_heads = 8
            input_size = 16

            self.transformer_d0 = MultiHeadTransformerPreLN(
                input_size=input_size, drop_rate=self.drop_rate, num_heads=num_heads,
            )
            self.transformer_d = MultiHeadTransformerPreLN(
                input_size=input_size, drop_rate=self.drop_rate, num_heads=num_heads,
            )

            pick_attentions = [nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads) for _ in range(self.classes)]
            self.pick_attentions = nn.ModuleList(pick_attentions)

    
    def forward(self, x, logits=False):
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        # Shared encoder part
        x = self.encoder(x)
        x = self.res_cnn_stack(x)
        x = self.bi_lstm_stack(x)
        x, _ = self.transformer_d0(x)
        x, _ = self.transformer_d(x)

        # Detection part
        detection = self.decoder_d(x)
        if logits:
            detection = self.conv_d(detection)
        else:
            detection = torch.sigmoid(self.conv_d(detection))
        detection = torch.squeeze(detection, dim=1)  # Remove channel dimension

        outputs = [detection]

        # Pick parts
        for lstm, attention, decoder, conv in zip(
            self.pick_lstms, self.pick_attentions, self.pick_decoders, self.pick_convs
        ):
            px = x.permute(
                2, 0, 1
            )  # From batch, channels, sequence to sequence, batch, channels
            px = lstm(px)[0]
            px = self.dropout(px)
            px = px.permute(1, 2, 0)  # From sequence, batch, channels to batch, channels, sequence
            
            if self.modified:
                px = px.permute(2, 0, 1)
                px, _ = attention(px, px, px)
                px = px.permute(1, 2, 0)
            else:
                px, _ = attention(px)

            px = decoder(px)
            if logits:
                pred = conv(px)
            else:
                pred = torch.sigmoid(conv(px))
            pred = torch.squeeze(pred, dim=1)  # Remove channel dimension

            outputs.append(pred)

        return tuple(outputs)

