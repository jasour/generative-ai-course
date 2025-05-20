# Transformer using  PyTorch's nn.TransformerEncoderLayer.
# ****************************************************************************************************************************************************************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------- Positional Encoding ---------------------------------------
def positional_encoding(max_len, d_model, device):
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ---------------- Scaled Dot-Product Attention ------------------------------
def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output

# ---------------- Functional Transformer Block ------------------------------
def transformer_block(x, w_q, w_k, w_v, w_o, linear1, linear2, norm1, norm2, dropout1, dropout2):
    B = x.size(1)
    q = torch.matmul(x, w_q)
    k = torch.matmul(x, w_k)
    v = torch.matmul(x, w_v)

    attn = scaled_dot_product_attention(q, k, v)
    attn_out = torch.matmul(attn, w_o)
    x = x + dropout1(attn_out)
    x = norm1(x)

    ff = linear2(F.relu(linear1(x)))
    x = x + dropout2(ff)
    x = norm2(x)
    return x

# ---------------- Build Transformer Encoder ---------------------------------
def build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward):
    layers = []
    head_dim = d_model // nhead
    for _ in range(num_layers):
        # Attention weights (for 1 head, can be extended for multi-head)
        w_q = nn.Parameter(torch.randn(d_model, d_model))
        w_k = nn.Parameter(torch.randn(d_model, d_model))
        w_v = nn.Parameter(torch.randn(d_model, d_model))
        w_o = nn.Parameter(torch.randn(d_model, d_model))

        linear1 = nn.Linear(d_model, dim_feedforward)
        linear2 = nn.Linear(dim_feedforward, d_model)
        norm1 = nn.LayerNorm(d_model)
        norm2 = nn.LayerNorm(d_model)
        dropout1 = nn.Dropout(0.1)
        dropout2 = nn.Dropout(0.1)

        def layer(x, wq=w_q, wk=w_k, wv=w_v, wo=w_o, l1=linear1, l2=linear2, n1=norm1, n2=norm2, d1=dropout1, d2=dropout2):
            return transformer_block(x, wq, wk, wv, wo, l1, l2, n1, n2, d1, d2)

        layers.append(layer)
    return layers
