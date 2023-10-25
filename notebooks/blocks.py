import torch.nn as nn
import torch.nn.functional as F
from seisbench.models.eqtransformer import *


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(SeparableConv1d, self).__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class EncoderSepConv1d(nn.Module):
    def __init__(self, input_channels, filters, kernel_sizes, in_samples):
        super().__init__()

        separable_convs = []
        pools = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            separable_convs.append(
                SeparableConv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size  //  2
                )
            )

            # To be consistent with the behaviour in tensorflow,
            # padding needs to be added for odd numbers of input_samples
            padding = in_samples % 2

            # Padding for MaxPool1d needs to be handled manually to conform with tf padding
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            in_samples = (in_samples + padding)  //  2

        self.separable_convs = nn.ModuleList(separable_convs)
        self.pools = nn.ModuleList(pools)

    def forward(self, x):
        for separable_convs, pool, padding in zip(self.separable_convs, self.pools, self.paddings):
            x = torch.relu(separable_convs(x))
            if padding != 0:
                # Only pad right, use -1e10 as negative infinity
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)

        return x


class DecoderSepConv1d(nn.Module):
    def __init__(self, input_channels, filters, kernel_sizes, out_samples, original_compatible=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.original_compatible = original_compatible

        # We need to trim off the final sample sometimes to get to the right number of output samples
        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding)  //  2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        separable_convs = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            separable_convs.append(
                SeparableConv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size  //  2
                )
            )

        self.separable_convs = nn.ModuleList(separable_convs)

    def forward(self, x):
        for i, separable_convs in enumerate(self.separable_convs):
            x = self.upsample(x)

            if self.original_compatible:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crops:
                    x = x[:, :, :-1]

            x = F.relu(separable_convs(x))

        return x


class DeeperBottleneckStack(nn.Module):
    def __init__(self, kernel_sizes, filters, drop_rate):
        super().__init__()

        members = []
        for ker in kernel_sizes:
            members.append(DeeperBottleneckBlock(filters, ker, drop_rate))

        self.members = nn.ModuleList(members)

    def forward(self, x):
        for member in self.members:
            x = member(x)

        return x


class DeeperBottleneckBlock(nn.Module):
    def __init__(self, filters, ker, drop_rate):
        super().__init__()

        self.manual_padding = False
        if ker == 3:
            padding = 1
        else:
            # ker == 2
            # Manual padding emulate the padding in tensorflow
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)

        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters//4, 1, bias=False)

        self.norm2 = nn.BatchNorm1d(filters//4, eps=1e-3)
        self.conv2 = nn.Conv1d(filters//4, filters//4, ker, stride=1, padding=padding, bias=False)

        self.norm3 = nn.BatchNorm1d(filters//4, eps=1e-3)
        self.conv3 = nn.Conv1d(filters//4, filters, 1, bias=False)

    def forward(self, x):
        residual = x
        y = self.norm1(x)
        y = F.relu(y)
        y = self.conv1(y)
        
        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)
        
        y = self.norm3(y)
        y = F.relu(y)
        y = self.conv3(y)

        # if self.stride !=1 or x.shape[1] != y.shape[1]:
        #     residual = self.conv3(x)
        #     residual = self.norm3(residual)
        
        y += residual

        return y

class MultiHeadTransformerPreLN(nn.Module):
    def __init__(self, input_size, drop_rate, num_heads, eps=1e-5):
        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=num_heads
        )
        self.norm1 = LayerNormalization(input_size)
        self.ff = FeedForward(input_size, drop_rate)
        self.norm2 = LayerNormalization(input_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm1(x)
        x = x.permute(2, 0, 1)
        y, weight = self.multihead_attention(x, x, x)
        y = x + y
        y = y.permute(0, 2, 1)
        y = self.norm2(y)
        y2 = self.ff(y)
        y2 = self.dropout(y2)
        y2 = y + y2
#         y2 = self.norm2(y2)

        return y2, weight

class MultiHeadTransformer(nn.Module):
    def __init__(self, input_size, drop_rate, num_heads, eps=1e-5):
        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=num_heads
        )
        self.norm1 = LayerNormalization(input_size)
        self.ff = FeedForward(input_size, drop_rate)
        self.norm2 = LayerNormalization(input_size)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, weight = self.multihead_attention(x, x, x)
        y = x + y
        y = y.permute(0, 2, 1)
        y = self.norm1(y)
        y2 = self.ff(y)
        y2 = y + y2
        y2 = self.norm2(y2)

        return y2, weight