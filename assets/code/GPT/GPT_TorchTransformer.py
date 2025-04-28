import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
import random

# ✅ Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # Token embeddings
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model))  # Positional embeddings

        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.fc_out = nn.Linear(d_model, vocab_size)  # Output layer
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Pass through Transformer Encoder-Decoder
        transformer_output = self.transformer(src_emb.permute(1, 0, 2), tgt_emb.permute(1, 0, 2))

        output = self.fc_out(transformer_output.permute(1, 0, 2))
        return self.softmax(output)


# ✅ Tokenize dataset correctly
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)


# ✅ Training Loop
def train(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        print(f"🔹 Starting Epoch {epoch+1}...")

        for step, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)

            # ✅ Reshape outputs & targets for loss computation
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 100 == 0:
                print(f"🔹 Epoch {epoch+1}, Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")


#  Text Generation Function
def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    output_ids = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(output_ids, output_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            output_ids = torch.cat([output_ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=True)



# ✅ Load GPT-2 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding

# ✅ Load AG News Dataset
dataset = load_dataset("ag_news", split="train")

# ✅ Reduce dataset size (Select 40,000 samples randomly)
subset_size = 10000
dataset = dataset.shuffle(seed=42).select(range(subset_size))

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ✅ Convert tokenized data to PyTorch tensors
input_ids = torch.tensor(tokenized_datasets["input_ids"], dtype=torch.long)
labels = torch.tensor(tokenized_datasets["input_ids"], dtype=torch.long)  # Labels are next-token predictions

# ✅ Create PyTorch DataLoader
dataset = TensorDataset(input_ids, labels)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

# ✅ Initialize Model & Training Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(vocab_size=50257).to(device)  # GPT-2 vocab size

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


if __name__ == "__main__":
    train(model, train_loader, optimizer, criterion, epochs=5)


# ✅ Example Usage
prompt_text = "The future of artificial intelligence is"
generated_text = generate_text(model, tokenizer, prompt_text)
print("🔹 Generated Text: ", generated_text)