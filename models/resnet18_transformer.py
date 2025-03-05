import torch
import torchvision
import math
import torch.nn as nn
# from CNN_Architectures.util import CBAM
from torch import Tensor
import CONFIG
from typing import Union


class ChannelAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Shared MLP
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        f = self.ca(x) * x
        f = self.sa(f) * f

        return f + x # Skip connection


class PositionalEncoding1D(nn.Module):
    """Classic Attention-is-all-you-need positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 300) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = self.make_pe(d_model, max_len)  # (d_model, 1, max_len)

    @staticmethod
    def make_pe(d_model: int, max_len: int) -> Tensor:
        """Compute positional encoding."""
        pe = torch.zeros(max_len, d_model).to(CONFIG.DEVICE)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(CONFIG.DEVICE)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(CONFIG.DEVICE)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (S, B, d_model)

        Returns:
            (B, d_model, H, W)
        """
        assert x.shape[2] == self.pe.shape[2], f"Input d_model ({x.shape[2]}) does not match positional encoding d_model ({self.pe.shape[2]})"  # type: ignore
        if x.size(0) > self.pe.size(0):  # Check if input length exceeds precomputed size
            self.pe = self.make_pe(self.pe.size(2), x.size(0)).to(x.device)

        x = x + self.pe[:x.size(0)]  # type: ignore
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    """2-D positional encodings for the feature maps produced by the encoder.

    Following https://arxiv.org/abs/2103.06450 by Sumeet Singh.

    Reference:
    https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/models/transformer_util.py
    """

    def __init__(self, d_model: int, max_h: int = 96, max_w: int = 384) -> None:
        super().__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, f"Embedding depth {d_model} is not even"
        pe = self.make_pe(d_model, max_h, max_w)  # (d_model, max_h, max_w)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(d_model: int, max_h: int = 96, max_w: int = 384) -> Tensor:
        """Compute positional encoding."""
        pe_h = PositionalEncoding1D.make_pe(d_model=d_model // 2, max_len=max_h)  # (max_h, 1 d_model // 2)
        pe_h = pe_h.permute(2, 0, 1).expand(-1, -1, max_w)  # (d_model // 2, max_h, max_w)

        pe_w = PositionalEncoding1D.make_pe(d_model=d_model // 2, max_len=max_w)  # (max_w, 1, d_model // 2)
        pe_w = pe_w.permute(2, 1, 0).expand(-1, max_h, -1)  # (d_model // 2, max_h, max_w)

        pe = torch.cat([pe_h, pe_w], dim=0)  # (d_model, max_h, max_w)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, d_model, H, W)

        Returns:
            (B, d_model, H, W)
        """
        assert x.shape[1] == self.pe.shape[0]  # type: ignore
        x = x + self.pe[:, : x.size(2), : x.size(3)]  # type: ignore
        return x


class ResNet18_Transformer(nn.Module):
    def __init__(self,
                 d_model: int,
                 dim_feedforward: int,
                 nhead: int,
                 dropout: float,
                 num_decoder_layers: int,
                 max_output_len: int,
                 sos_index: int,
                 eos_index: int,
                 pad_index: int,
                 num_classes: int,
                 ):
        super().__init__()
        self.d_model = d_model
        self.max_output_len = max_output_len + 2
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index

        # Encoder
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1,
            CBAM(64, 4),
            resnet.bn1,
            nn.ReLU(),
            resnet.maxpool,
            resnet.layer1,
            CBAM(64, 4),
            resnet.layer2,
            CBAM(128, 4),
            resnet.layer3,
            CBAM(256, ratio=8),
            resnet.layer4,
        )

        self.bottleneck = nn.Conv2d(512, d_model, 1)
        self.bottleneck_activation = nn.LeakyReLU(0.05, True)

        self.pos_encoder = PositionalEncoding2D(d_model)

        # Decoder
        self.target_embedding = nn.Embedding(num_classes, self.d_model)

        # Apply attention is all you need positional encoding
        self.word_positional_encoder = PositionalEncoding1D(self.d_model, dropout=dropout, max_len=self.max_output_len)
        transformer_decoder_layer = nn.TransformerDecoderLayer(self.d_model,
                                                               nhead=nhead,
                                                               dim_feedforward=dim_feedforward,
                                                               dropout=dropout,
                                                               activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_decoder_layers)
        self.fc = nn.Linear(self.d_model, num_classes)

        # Initialize weights if training
        if self.training:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        init_range = 0.1
        self.target_embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

        nn.init.kaiming_normal_(
            self.bottleneck.weight.data,
            a=0,
            mode="fan_out",
            nonlinearity="leaky_relu",
        )
        if self.bottleneck.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.bottleneck.weight.data)
            bound = 1 / math.sqrt(fan_out + 1e-9)
            nn.init.normal_(self.bottleneck.bias, -bound, bound)

    def forward(self, x, y):
        """
        Args:
            x: (B, _E, _H, _W)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (B, num_classes, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)

        output = self.decode(y, encoded_x)  # (Sy, B, num_classes)
        output = output.permute(1, 2, 0)  # (B, num_classes, Sy)
        return output

    def encode(self, x):
        """Encode inputs.
        Args:
            x: (B, C, _H, _W)
        Returns:
            (Sx, B, E)
        """
        # Resnet expects 3 channels but training images are in gray scale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.backbone(x)  # (B, RESNET_DIM, H, W); H = _H // 32, W = _W // 32

        x = self.bottleneck(x)  # (B, d_model, H, W)
        x = self.bottleneck_activation(x)

        x = self.pos_encoder(x)  # (B, H, W, hidden_size * 2)
        x = x.flatten(start_dim=2)

        # Permute to (Sx, B, E) where Sx = H * W
        x = x.permute(2, 0, 1)  # (Sx, B, E)

        return x

    def decode(self, y, encoded_x):
        """Decode encoded inputs with teacher-forcing.

        Args:
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (Sy, B, num_classes) logits
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.target_embedding(y) * math.sqrt(self.d_model)  # (Sy, B, E)
        y = self.word_positional_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]

        y_mask = generate_square_subsequent_mask(Sy).type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)

        output = self.fc(output)  # (Sy, B, num_classes)
        return output

    def greedy_search(self, x):
        """Greedy Search decoding for sequence generation.

        Args:
            x: (B, C, H, W). Input images.

        Returns:
            (B, max_output_len) with elements in (0, num_classes - 1).
        """
        B = x.shape[0]
        S = self.max_output_len

        with torch.no_grad():  # Prevents tracking gradients
            encoded_x = self.encode(x)

            output_indices = torch.full((B, S), self.pad_index).type_as(x).long()
            output_indices[:, 0] = self.sos_index

            print(output_indices)

            has_ended = torch.full((B,), False)

            for Sy in range(1, S):
                y = output_indices[:, :Sy]  # (B, Sy)
                logits = self.decode(y, encoded_x)  # (Sy, B, num_classes)

                # Select the token with the highest conditional probability
                output = torch.argmax(logits, dim=-1)  # (Sy, B)


                output_indices[:, Sy] = output[-1:]  # Set the last output token

                # Early stopping of prediction loop to speed up prediction
                has_ended |= (output_indices[:, Sy] == self.eos_index).type_as(has_ended)
                if torch.all(has_ended):
                    break

            # Set all tokens after end token to be padding
            eos_positions = find_first(output_indices, self.eos_index)
            for i in range(B):
                j = int(eos_positions[i].item()) + 1
                output_indices[i, j:] = self.pad_index

        return output_indices


def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask.to('cuda')


def find_first(x: Tensor, element: Union[int, float], dim: int = 1):
    """Find the first occurence of element in x along a given dimension.

        Args:
            x: The input tensor to be searched.
            element: The number to look for.
            dim: The dimension to reduce.

        Returns:
            Indices of the first occurrence of the element in x. If not found, return the
            length of x along dim.

        Usage:
            first_element(Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
            tensor([2, 1, 3])

        Reference:
            https://discuss.pytorch.org/t/first-nonzero-index/24769/9

            I fixed an edge case where the element we are looking for is at index 0. The
            original algorithm will return the length of x instead of 0.
        """
    mask = x == element
    found, indices = ((mask.cumsum(dim) == 1) & mask).max(dim)
    indices[(~found) & (indices == 0)] = x.shape[dim]

    return indices

