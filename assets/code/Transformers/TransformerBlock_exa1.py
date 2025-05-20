
# Transformer using  PyTorch's nn.TransformerEncoderLayer.
# ****************************************************************************************************************************************************************************************************
def positional_encoding(max_len, d_model, device): 
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, device=device).unsqueeze(1) # constructing a column vector of integer positions, which represent the index of each token (or input element) in the sequence. e.g. max_len=2: tensor([[0],[1]], device='mps:0')
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)) # creates a vector of frequency scaling terms. e.g., d_model =4, tensor([1.0000, 0.0100], device='mps:0')
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # Tensor of shape: (max_len, d_model), e.g., tensor([   [0.0000, 1.0000, 0.0000, 1.0000] ,  [0.8415, 0.5403, 0.0100, 1.0000]  ], device='mps:0')

# ---------------- Create the Transformer Encoder (A single Transformer block) ----------------------------
def build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward): 
    layers = []
    for _ in range(num_layers):
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        layers.append(layer)
    return nn.Sequential(*layers)


# ---------------- Setup Model Parts -----------------------------------------
def build_model(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, device="cpu"):
    input_proj = nn.Linear(1, d_model).to(device) # Projects a scalar input (1 value) into a d_model-dimensional vector.
    transformer = build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward)
    transformer.to(device)
    pos_enc = positional_encoding(2, d_model, device=device)
    output_proj = nn.Sequential(   # After the Transformer processes two vectors of size d_model, we flatten them into a single vector of size 2 * d_modelf followed by a MLP
        nn.Linear(2 * d_model, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    return input_proj, transformer, pos_enc, output_proj

