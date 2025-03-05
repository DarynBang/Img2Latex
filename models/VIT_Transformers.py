import torch
import torchvision
import math
import torch.nn as nn
from torch import Tensor
import CONFIG
from typing import Union

from math import ceil
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Helper functions:

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args, **kwargs: val

def cast_tuple(val, l=3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))


class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
             nn.Conv2d(dim, dim * mult, 1, bias=False),
             nn.BatchNorm2d(dim * mult),
             nn.Hardswish(),
             nn.Dropout(dropout),
             nn.Conv2d(dim * mult, dim, 1),
             nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, fmap_h, fmap_w, num_heads=8, dim_key=64, dim_value=64, dropout=0., dim_out=None, downsample=False):
        super().__init__()
        inner_dim_key = dim_key * num_heads
        inner_dim_value = dim_value * num_heads
        dim_out = default(dim_out, dim)

        self.num_heads = num_heads
        self.scale = dim_key ** -0.5

        self.to_query = nn.Sequential(
            nn.Conv2d(dim, inner_dim_key, 1, stride=(2 if downsample else 1), bias=False),
            nn.BatchNorm2d(inner_dim_key)
        )
        self.to_key = nn.Sequential(
            nn.Conv2d(dim, inner_dim_key, 1, stride=1, bias=False),
            nn.BatchNorm2d(inner_dim_key)
        )
        self.to_value = nn.Sequential(
            nn.Conv2d(dim, inner_dim_value, 1, stride=1, bias=False),
            nn.BatchNorm2d(inner_dim_value)
        )

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        self.f_out = nn.Sequential(
            nn.Hardswish(),
            nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm,
            nn.Dropout(dropout)
        )

        # Positional bias
        self.positional_bias = nn.Embedding(fmap_h * fmap_w, num_heads)

        q_range_h = torch.arange(0, fmap_h, step=(2 if downsample else 1))
        q_range_w = torch.arange(0, fmap_w, step=(2 if downsample else 1))
        k_range_h = torch.arange(0, fmap_h)
        k_range_w = torch.arange(0, fmap_w)

        q_pos = torch.stack(torch.meshgrid(q_range_h, q_range_w, indexing='ij'), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range_h, k_range_w, indexing='ij'), dim=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'h w c -> (h w) c'), (q_pos, k_pos))
        relative_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_relative, y_relative = relative_pos.unbind(dim=-1)
        pos_indices = (x_relative * fmap_w) + y_relative  # Use fmap_w for width-based indexing

        self.register_buffer('pos_indices', pos_indices)


    def apply_pos_bias(self, fmap):
        bias = self.positional_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)


    def forward(self, x):
        q = self.to_query(x)

        h = q.shape[2]
        w = q.shape[3]

        k = self.to_key(x)
        v = self.to_value(x)

        qkv = (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.num_heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h=self.num_heads, x=h, y=w)

        return self.f_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, fmap_h, fmap_w, depth, num_heads, dim_key, dim_value, mlp_mult=2, dropout=0., dim_out=None, downsample=False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and dim == dim_out

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim,
                          fmap_h=fmap_h,
                          fmap_w=fmap_w,
                          num_heads=num_heads,
                          dim_key=dim_key,
                          dim_value=dim_value,
                          dropout=dropout,
                          downsample=downsample,
                          dim_out=dim_out),
                FeedForward(dim_out, mlp_mult, dropout),
                ]))


    def forward(self, x):
        for attn, ff in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x) + attn_res
            x = ff(x) + x

        return x


class ConvPatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, out_channels, 3, stride=2, padding=1)

        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=16)

    def forward(self, x):
        res = self.residual(x)  # Downsample input directly
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x + res  # Add residual connection


class LeVisionTransformer(nn.Module):
    def __init__(self,
                 img_size,
                 num_classes,
                 dim,
                 depth,
                 num_heads,
                 mlp_mult,
                 stages=2,
                 dim_key=64,
                 dim_value=64,
                 dropout=0.,
                 num_distill_classes=None):
        super().__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(num_heads, stages)
        self.num_classes = num_classes

        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and num_heads must be a tuple that is less than the designated number of stages'
        super().__init__()

        self.conv_patch_embedding = ConvPatchEmbedding(3, dims[0])

        fmap_h = img_size[0] // (2 ** 4)
        fmap_w = img_size[1] // (2 ** 4)

        layers = []

        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == (stages - 1)

            layers.append(
                Transformer(dim, fmap_h, fmap_w, 1, heads * 2, dim_key, dim_value, mlp_mult, dropout)
            )

            if not is_last:
                next_dim = dims[ind + 1]
                layers.append(
                    Transformer(dim, fmap_h, fmap_w, 1, heads * 2, dim_key, dim_value, dim_out=next_dim, downsample=True)
                )
                fmap_h = ceil(fmap_h / 2)
                fmap_w = ceil(fmap_w / 2)

        self.backbone = nn.Sequential(*layers)

        if num_classes > 0:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Rearrange('... () () -> ...')
            )

        else:   # If used for feature extraction -> (batch_size, num_patches, feature_dim),
            self.pool = Rearrange('b c h w -> b (h w) c')  # Flatten spatial dimensions

        self.distill_head = nn.Linear(dim, num_distill_classes) if exists(num_distill_classes) else always(None)
        self.mlp_head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else None

    def forward(self, img):
        x = self.conv_patch_embedding(img)

        x = self.backbone(x)

        x = self.pool(x)

        if self.num_classes == 0:
            return x

        out = self.mlp_head(x)
        distill = self.distill_head(x) if exists(self.distill_head) else None

        if exists(distill):
            return out, distill

        return out


class PositionalEncoding1D(nn.Module):
    """Classic Attention-is-all-you-need positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_encoder = self.make_pe(d_model, max_len)  # (max_len, 1, d_model)

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
        assert x.shape[2] == self.pos_encoder.shape[
            2], f"Input d_model ({x.shape[2]}) does not match positional encoding d_model ({self.pos_encoder.shape[2]})"  # type: ignore
        if x.size(0) > self.pos_encoder.size(0):  # Check if input length exceeds precomputed size
            self.pos_encoder = self.make_pe(self.pos_encoder.size(2), x.size(0)).to(x.device)

        x = x + self.pos_encoder[:x.size(0)]  # type: ignore
        return self.dropout(x)


class FeatureExtractor(nn.Module):
    def __init__(self, img_size, patch_size, vit_emb, depth=8, nheads=8, mlp_dim=1024, out_dim=1024, dropout=0.2):
        super().__init__()

        # # ResNet Backbone (Truncated)
        # resnet = torchvision.models.resnet34(weights=None)
        # self.resnetLayers = nn.Sequential(
        #     resnet.conv1,
        #     resnet.bn1,
        #     nn.ReLU(True),
        #     resnet.maxpool,
        #     resnet.layer1,
        #     resnet.layer2,
        #     resnet.layer3,
        # )

        # # Global Average Pooling for ResNet Features
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Reduces to [B, C, 1, 1]
        # self.resnet_fc = nn.Linear(256, out_dim)  # Projects ResNet output to match ViT

        # Vision Transformer Backbone
        self.LeVIT = LeVisionTransformer(
            img_size=img_size,
            num_classes=0,
            dim=vit_emb,
            depth=6,
            num_heads=nheads,
            mlp_mult=4,
            stages=2,
            dropout=dropout
        )

    def forward(self, x):
        # Extract ResNet Features
        # resnet_f = self.resnetLayers(x)  # [B, 256, H', W']
        # resnet_f = self.global_avg_pool(resnet_f).squeeze(-1).squeeze(-1)  # [B, 256]
        # resnet_f = self.resnet_fc(resnet_f)  # [B, out_dim]
        # resnet_f = resnet_f.unsqueeze(1)  # [B, 1, out_dim] (to match ViT format)

        # Extract ViT Features
        levit_f = self.LeVIT(x)  # [B, Seq_Len, vit_emb]

        # Concatenate Along Sequence Length Dimension
        # f = torch.cat([resnet_f, vit_f], dim=1)  # [B, Seq_Len+1, out_dim]

        return levit_f

class ViT_Transformer(nn.Module):
    def __init__(self,
                 img_size: tuple[int, int],
                 num_encoder_layers: int,
                 vit_emb: int,
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
                 training: bool = True
                 ):
        super().__init__()
        self.d_model = d_model
        self.max_output_len = max_output_len + 2
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.training = training

        # Encoder
        self.backbone = LeVisionTransformer(
            img_size=img_size,
            num_classes=0,
            dim=vit_emb,
            depth=num_encoder_layers,
            num_heads=nhead,
            mlp_mult=4,
            stages=2,
            dropout=dropout
        )

        self.bottleneck = nn.Linear(vit_emb, d_model)
        self.bottleneck_activation = nn.LeakyReLU(0.05)
        self.norm = nn.LayerNorm(d_model)

        # Decoder
        self.target_embedding = nn.Embedding(num_classes, self.d_model)

        # Apply attention is all you need positional encoding
        self.word_positional_encoder = PositionalEncoding1D(self.d_model, max_len=self.max_output_len)
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

        if self.bottleneck.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.bottleneck.weight.data)
            bound = 1 / (math.sqrt(fan_out) + 1e-9)
            nn.init.normal_(self.bottleneck.bias, -bound, bound)

    def forward(self, x, y, memory_key_padding_mask=None):
        """
        Args:
            x: (B, _E, _H, _W)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (B, num_classes, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)

        output = self.decode(y, encoded_x, memory_key_padding_mask)  # (Sy, B, num_classes)
        output = output.permute(1, 2, 0)  # (B, num_classes, Sy)
        return output

    def encode(self, x):
        """Encode inputs.
        Args:
            x: (B, C, _H, _W)
        Returns:
            (Sx, B, E)
        """
        # VIT expects 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.backbone(x)  # (batch_size, num_patches, feature_dim)
        x = self.bottleneck(x)
        x = self.bottleneck_activation(x)
        x = self.norm(x)

        # Permute to (Sx, B, E) where Sx = H * W
        x = x.permute(1, 0, 2)  # (Sx, B, E)

        return x

    def decode(self, y, encoded_x, memory_key_padding_mask=None):
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

        # Generate the causal mask for decoding
        y_mask = torch.full((Sy, Sy), float("-inf"), device=y.device)
        y_mask.triu_(1)  # Fill upper triangle with -inf (excluding diagonal)

        output = self.transformer_decoder(
            tgt=y,
            memory=encoded_x,
            tgt_mask=y_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # (Sy, B, E)

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

        encoded_x = self.encode(x)

        output_indices = torch.full((B, S), self.pad_index).type_as(x).long()
        output_indices[:, 0] = self.sos_index
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

    def beam_search(self, x, k_beams=3, early_stopping=True):
        """Beam Search decoding for sequence generation.

        Args:
            x: (B, C, H, W). Input images.

        Returns:
            (B, max_output_len) with elements in (0, num_classes - 1).
        """
        B = x.shape[0]
        device = x.device
        S = self.max_output_len

        encoded_x = self.encode(x)

        # Intializing beams
        sequences = [[[self.sos_index], 0.0]] * k_beams  # each beam: [[tokens], score]

        for _ in range(S):
            all_candidates = []

            for sequence, score in sequences:
                if sequence[-1] == self.eos_index:
                    all_candidates.append((sequence, score))
                    continue

                y = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)  # (1, Sy)
                logits = self.decode(y, encoded_x)  # (Sy, B, num_classes)
                log_probs = nn.functional.log_softmax(logits[-1], dim=-1)  # (B, num_classes)

                top_k_probs, top_k_indices = torch.topk(log_probs, k_beams, dim=-1)  # (B, k_beams)

                for i in range(k_beams):
                    candidate_sequence = sequence + [top_k_indices[0, i].item()]
                    candidate_score = score + top_k_probs[0, i].item()

                    all_candidates.append((candidate_sequence, candidate_score))

            # Select k_beams best sequences
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:k_beams]

            # Early stopping
            if early_stopping and all(seq[-1] == self.eos_index for seq, _ in sequences):
                break

        # Select the best sequence
        best_seq = max(sequences, key=lambda tup: tup[1])[0]

        # Pad the sequence to max_output_len
        output = torch.full((B, S), self.pad_index, device=device, dtype=torch.long)
        output[0, :len(best_seq)] = torch.tensor(best_seq, device=device)

        return output


def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask.to('cuda:0')


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

