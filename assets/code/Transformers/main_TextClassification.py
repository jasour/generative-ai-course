'''
ViT vs Text Transformer: Structural Similarities: 
                            ViT (Vision)                   Vs        Text Classification
Input sequence:   Flattened image patches                  Vs    Sequence of word/token embeddings
Token embeddings: Patch → Linear projection                VS    Token → Embedding lookup
Positional encoding:   Add to patch embeddings             Vs     Add to word embeddings
[CLS] token:  Used to aggregate image info                 Vs     Used to aggregate sentence meaning
Transformer layers:  Encode interactions between patches   Vs    Encode word-level relationships
Output head:   MLP on [CLS] → digit class                  Vs   MLP on [CLS] → text class (e.g., spam/not spam)

The AG_NEWS dataset has 4 categories: 0: World, 1: Sports, 2: Business, 3: Science/Technology.
'''



# ------------------------- Transformer for Text Classification -------------------------
# Token-based Transformer encoder for classifying text sequences using a CLS token

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import random
from torchtext.datasets import AG_NEWS
train_iter, test_iter = AG_NEWS(root='.data')

# ---------------- Positional Encoding ---------------------------------------
def positional_encoding(max_len, d_model, device):
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ---------------- Transformer Encoder Builder -------------------------------
def build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward):
    layers = []
    for _ in range(num_layers):
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        layers.append(layer)
    return nn.Sequential(*layers)

# ---------------- Load IMDB Dataset ------------------------------------------
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter, OrderedDict

# Tokenizer and vocab setup
tokenizer = get_tokenizer("basic_english")
from collections import Counter
print("🔍 Class distribution in AG_NEWS training set:")
print(Counter([label for label, _ in train_iter]))
train_iter, test_iter = AG_NEWS(root=".data")  # Reload it after counting

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(str(text))

# Build vocabulary from training set
#train_iter = AG_NEWS(split='train')
# Build vocabulary from training text
counter = Counter()
for tokens in yield_tokens(train_iter):
    counter.update(tokens)

# Sort and add specials
specials = ["<pad>", "<unk>"]
vocab = Vocab(counter, specials=specials)
text_pipeline = lambda x: [vocab.stoi.get(token, vocab.stoi["<unk>"]) for token in tokenizer(x)]

# Collate function to pad and convert text to tensor
def collate_batch(batch):
    text_pipeline = lambda x: [vocab.stoi.get(tok, vocab.stoi["<unk>"]) for tok in tokenizer(str(x))]
    label_pipeline = lambda x: int(x) - 1
    labels, texts = zip(*batch)
    token_ids = [torch.tensor(text_pipeline(text)[:128]) for text in texts]
    labels = torch.tensor([label_pipeline(label) for label in labels])
    padded = nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=vocab["<pad>"])
    return padded, labels

# Dataloaders
train_iter, test_iter = AG_NEWS()
from torch.utils.data import DataLoader
train_loader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=32, collate_fn=collate_batch)

# ---------------- Build Text Transformer Model ------------------------------
def build_model(vocab_size, seq_len, num_classes, d_model, nhead, num_layers, dim_feedforward, device):
    token_embed = nn.Embedding(vocab_size, d_model).to(device)
    transformer = build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward).to(device)
    pos_enc = positional_encoding(seq_len + 1, d_model, device=device)  # +1 for CLS
    cls_token = nn.Parameter(torch.zeros(1, 1, d_model, device=device))
    output_proj = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, num_classes)
    ).to(device)
    return token_embed, transformer, pos_enc, cls_token, output_proj

# ---------------- Transformer Forward Pass ----------------------------------
def transformer_forward(x, token_embed, transformer, pos_enc, cls_token, output_proj):
    B, T = x.shape  # (batch_size, sequence_length)
    x = token_embed(x)  # (B, T, d_model)
    cls_tok = cls_token.expand(-1, B, -1)  # (1, B, d_model)
    x = torch.cat([cls_tok, x.permute(1, 0, 2)], dim=0)  # (T+1, B, d_model)
    x = x + pos_enc[:x.size(0)].unsqueeze(1)
    for layer in transformer:
        x = layer(x)
    return output_proj(x[0])  # Use CLS token output (B, num_classes)

# ---------------- Training and Evaluation -----------------------------------
def train_and_eval(device):
    # Hyperparameters
    vocab_size = len(vocab)
    seq_len = 128
    num_classes = 4
    d_model = 64 # 128
    nhead = 4
    num_layers = 4
    dim_feedforward = 128 # 256
    batch_size = 32
    num_epochs = 10
    lr = 1e-3

    # Build model
    token_embed, transformer, pos_enc, cls_token, output_proj = build_model(
        vocab_size, seq_len, num_classes, d_model, nhead, num_layers, dim_feedforward, device)

    params = list(token_embed.parameters()) + list(output_proj.parameters()) + [cls_token]
    for layer in transformer:
        params += list(layer.parameters())

    optimizer = optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        token_embed.train(), transformer.train(), output_proj.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            logits = transformer_forward(xb, token_embed, transformer, pos_enc, cls_token, output_proj)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}")
    
    # Save model parts
    torch.save({
        'token_embed': token_embed.state_dict(),
        'transformer': [layer.state_dict() for layer in transformer],
        'output_proj': output_proj.state_dict(),
        'cls_token': cls_token,
        'pos_enc': pos_enc,
        'vocab': vocab.stoi  # just storing stoi (can reconstruct vocab later)
        }, "text_transformer_model.pth")
    print("Model saved to text_transformer_model.pth")

    # Evaluation
    token_embed.eval(), transformer.eval(), output_proj.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            logits = transformer_forward(xb, token_embed, transformer, pos_enc, cls_token, output_proj)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Test Accuracy: {100. * correct / total:.2f}%")
    return token_embed, transformer, pos_enc, cls_token, output_proj


def predict_text(text, model_parts, device, seq_len=128):
    token_embed, transformer, pos_enc, cls_token, output_proj = model_parts
    model_input = torch.tensor(
        [vocab.stoi.get(tok, vocab.stoi["<unk>"]) for tok in tokenizer(text)[:seq_len]],
        dtype=torch.long
    ).unsqueeze(0).to(device)  # Shape: (1, seq_len)

    model_input = nn.functional.pad(
        model_input, (0, seq_len - model_input.size(1)), value=vocab.stoi["<pad>"]
    )  # Pad to seq_len

    token_embed.eval(), transformer.eval(), output_proj.eval()
    with torch.no_grad():
        x = token_embed(model_input)  # (1, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, 1, d_model)

        cls_tok = cls_token.expand(-1, 1, -1)  # (1, 1, d_model)
        x = torch.cat([cls_tok, x], dim=0)  # (seq_len+1, 1, d_model)

        x = x + pos_enc[:x.size(0)].unsqueeze(1)

        for layer in transformer:
            x = layer(x)

        logits = output_proj(x[0])
        probs = torch.softmax(logits, dim=1)
        print("Probabilities:", probs.cpu().numpy())  # 🔍 Show confidence for each class
        pred = logits.argmax(dim=1).item()
    return pred


# ---------------- Run Script ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_parts = train_and_eval(device)

sample_text = "Stocks rallied after the Federal Reserve held interest rates."
pred_label = predict_text(sample_text, model_parts, device)
print(f"Predicted class: {pred_label}")