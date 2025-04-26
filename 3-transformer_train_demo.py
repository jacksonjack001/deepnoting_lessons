import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# --- 1. 超参数与配置 ---
# (为了参数量 < 10k，这些值必须非常小)
SRC_LANG = "en"
TGT_LANG = "fr"
D_MODEL = 32  # 嵌入维度 (非常小)
NHEAD = 2  # 注意力头数 (非常小)
NUM_ENCODER_LAYERS = 1  # 编码器层数 (非常小)
NUM_DECODER_LAYERS = 1  # 解码器层数 (非常小)
DIM_FEEDFORWARD = 64  # FFN内部维度 (通常是 d_model 的 2-4 倍)
DROPOUT = 0.1
MAX_LEN = 10  # 序列最大长度 (足够用于数字)
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. 准备简单数据集 ---
# 英文到法文的数字 (少量训练数据)
raw_data = [
    ("one", "un"),
    ("two", "deux"),
    ("three", "trois"),
    ("four", "quatre"),
    ("five", "cinq"),
    ("six", "six"),
    ("seven", "sept"),
    ("eight", "huit"),
    ("nine", "neuf"),
    ("ten", "dix"),
    ("eleven", "onze"),
    ("twelve", "douze"),
    ("thirteen", "treize"),
    ("fourteen", "quatorze"),
    ("fifteen", "quinze"),
    # 可以添加更多，但保持简单
]

arr = list("abcdef")
raw_data = [(str(a), b) for a, b in zip(range(6), arr)]

# 添加特殊标记
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"  # Start Of Sentence
EOS_TOKEN = "<eos>"  # End Of Sentence
UNK_TOKEN = "<unk>"  # Unknown token


# --- 3. 构建词表 ---
def build_vocab(texts, lang):
    counter = Counter()
    for text in texts:
        counter.update(text.split(" "))  # 简单按空格分词
    # 创建词表，包含特殊标记
    vocab = {
        token: i
        for i, token in enumerate(
            [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(counter.keys())
        )
    }
    return vocab, {
        i: token for token, i in vocab.items()
    }  # 返回词到索引和索引到词的映射


src_texts = [pair[0] for pair in raw_data]
tgt_texts = [pair[1] for pair in raw_data]

SRC_VOCAB, SRC_ITOS = build_vocab(src_texts, SRC_LANG)
TGT_VOCAB, TGT_ITOS = build_vocab(tgt_texts, TGT_LANG)

SRC_VOCAB_SIZE = len(SRC_VOCAB)
TGT_VOCAB_SIZE = len(TGT_VOCAB)
PAD_IDX = SRC_VOCAB[PAD_TOKEN]  # 假设源和目标使用相同的 PAD 索引

print(f"Source Vocab Size: {SRC_VOCAB_SIZE}")
print(f"Target Vocab Size: {TGT_VOCAB_SIZE}")
print(f"PAD index: {PAD_IDX}")


# --- 4. 数据预处理 ---
def tokenize_and_numericalize(text, vocab):
    tokens = text.split(" ")
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]


data = []
for src, tgt in raw_data:
    src_indices = (
        [SRC_VOCAB[SOS_TOKEN]]
        + tokenize_and_numericalize(src, SRC_VOCAB)
        + [SRC_VOCAB[EOS_TOKEN]]
    )
    tgt_indices = (
        [TGT_VOCAB[SOS_TOKEN]]
        + tokenize_and_numericalize(tgt, TGT_VOCAB)
        + [TGT_VOCAB[EOS_TOKEN]]
    )
    data.append(
        (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(tgt_indices, dtype=torch.long),
        )
    )


# --- 5. 数据集和数据加载器 ---
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    # 填充序列
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


train_dataset = TranslationDataset(data)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)


# --- 6. 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(
            0, 1
        )  # Shape: [max_len, 1, d_model] -> PyTorch Transformer expects [seq_len, batch, dim] if batch_first=False
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model] (if batch_first=False)
        # or [batch_size, seq_len, d_model] (if batch_first=True)
        # The nn.Transformer module can handle both if specified correctly.
        # Here, assuming batch_first=False for PE compatibility if nn.Transformer uses it.
        # If nn.Transformer(batch_first=True), PE needs adjustment or input needs transpose.
        # Let's adapt for batch_first=True which is more intuitive.
        # x shape: [batch_size, seq_len, d_model]
        # pe shape needs to be broadcastable: [1, max_len, d_model]
        pe_for_batch = (
            self.pe[: x.size(1), :].squeeze(1).unsqueeze(0)
        )  # shape [1, seq_len, d_model]
        x = x + pe_for_batch  # Add positional encoding
        return self.dropout(x)


# --- 7. Transformer 模型 ---
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=MAX_LEN)

        # Using PyTorch's Transformer module
        # IMPORTANT: Set batch_first=True for easier tensor manipulation
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Set batch_first to True
        )

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _create_padding_mask(self, seq, pad_idx):
        # seq shape: [batch_size, seq_len]
        # output shape: [batch_size, seq_len]
        return seq == pad_idx

    def forward(self, src, tgt):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]

        # Create masks
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # Decoder target mask (causal mask)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(
            DEVICE
        )  # [tgt_len, tgt_len]

        # Padding masks
        src_padding_mask = self._create_padding_mask(src, PAD_IDX).to(
            DEVICE
        )  # [batch_size, src_len]
        tgt_padding_mask = self._create_padding_mask(tgt, PAD_IDX).to(
            DEVICE
        )  # [batch_size, tgt_len]

        # Embedding + Positional Encoding
        # Input shape for embedding: [batch_size, seq_len]
        # Output shape after embedding: [batch_size, seq_len, d_model]
        src_emb = self.positional_encoding(
            self.src_tok_emb(src) * math.sqrt(self.d_model)
        )
        tgt_emb = self.positional_encoding(
            self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        )

        # Transformer forward pass
        # Input shapes (batch_first=True):
        # src: [batch_size, src_len, d_model]
        # tgt: [batch_size, tgt_len, d_model]
        # Masks shapes:
        # tgt_mask: [tgt_len, tgt_len] (Transformer handles broadcasting to heads)
        # src_padding_mask: [batch_size, src_len]
        # tgt_padding_mask: [batch_size, tgt_len]
        # memory_key_padding_mask: [batch_size, src_len] (same as src_padding_mask)

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,  # Use src padding for memory
        )
        # Output shape: [batch_size, tgt_len, d_model]

        # Final linear layer to get logits
        # Input shape: [batch_size, tgt_len, d_model]
        # Output shape: [batch_size, tgt_len, tgt_vocab_size]
        return self.generator(output)


# --- 8. 初始化模型和优化器 ---
model = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    D_MODEL,
    NHEAD,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    DIM_FEEDFORWARD,
    DROPOUT,
).to(DEVICE)


# --- 9. 计算参数量 ---
def count_parameters(model):
    arr = [p.numel() for p in model.parameters() if p.requires_grad]
    print(arr)
    return sum(arr)


num_params = count_parameters(model)
print(f"The model has {num_params:,} trainable parameters.")
if num_params >= 10000:
    print(
        "WARNING: Parameter count is >= 10,000. Adjust hyperparameters (D_MODEL, NHEAD, LAYERS, DIM_FEEDFORWARD) further if needed."
    )
else:
    print("Parameter count is below 10,000.")

# --- 10. 损失函数和优化器 ---
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 忽略填充标记的损失
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# --- 11. 训练循环 ---
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    start_time = time.time()

    for src, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # Prepare target input and output for teacher forcing
        # Target input: <sos> w1 w2 ... wn
        # Target output: w1 w2 ... wn <eos>
        tgt_input = tgt[:, :-1]  # Exclude <eos>
        tgt_output = tgt[:, 1:]  # Exclude <sos>

        optimizer.zero_grad()

        # Forward pass
        logits = model(src, tgt_input)  # Shape: [batch_size, tgt_len-1, tgt_vocab_size]

        # Calculate loss
        # Reshape for CrossEntropyLoss:
        # Logits: [batch_size * (tgt_len-1), tgt_vocab_size]
        # Target: [batch_size * (tgt_len-1)]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    end_time = time.time()
    epoch_loss = total_loss / len(dataloader)
    epoch_time = end_time - start_time
    print(f"Epoch Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
    return epoch_loss


print("\nStarting Training...")
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"--- Epoch {epoch} ---")
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion)

print("Training Finished!")


# --- 12. 翻译（推理）函数 ---
def translate(
    model,
    src_sentence,
    src_vocab,
    tgt_vocab,
    src_itos,
    tgt_itos,
    device,
    max_len=MAX_LEN,
):
    model.eval()  # Set model to evaluation mode

    # Tokenize and numericalize source sentence
    src_tokens = (
        [src_vocab[SOS_TOKEN]]
        + tokenize_and_numericalize(src_sentence, src_vocab)
        + [src_vocab[EOS_TOKEN]]
    )
    src_tensor = (
        torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    )  # Add batch dimension

    # Create source padding mask (no padding needed for single sentence, but good practice)
    src_padding_mask = model._create_padding_mask(src_tensor, PAD_IDX).to(device)

    # Encode the source sentence
    src_emb = model.positional_encoding(
        model.src_tok_emb(src_tensor) * math.sqrt(model.d_model)
    )
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
    # memory shape: [1, src_len, d_model]

    # Start decoding with <sos> token
    tgt_tokens = [tgt_vocab[SOS_TOKEN]]
    eos_idx = tgt_vocab[EOS_TOKEN]

    for i in range(max_len - 1):
        tgt_tensor = (
            torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)
        )  # [1, current_tgt_len]

        # Create masks for the decoder
        tgt_seq_len = tgt_tensor.shape[1]
        tgt_mask = model._generate_square_subsequent_mask(tgt_seq_len).to(
            device
        )  # [current_tgt_len, current_tgt_len]
        tgt_padding_mask = model._create_padding_mask(tgt_tensor, PAD_IDX).to(
            device
        )  # [1, current_tgt_len]

        # Decode one step
        tgt_emb = model.positional_encoding(
            model.tgt_tok_emb(tgt_tensor) * math.sqrt(model.d_model)
        )
        output = model.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,  # Use the source padding mask for memory
        )
        # output shape: [1, current_tgt_len, d_model]

        # Get logits for the *last* token prediction
        last_token_logits = model.generator(
            output[:, -1, :]
        )  # shape [1, tgt_vocab_size]

        # Find the token with the highest probability (greedy decoding)
        pred_token_idx = last_token_logits.argmax(1).item()
        tgt_tokens.append(pred_token_idx)

        # Stop if <eos> is predicted
        if pred_token_idx == eos_idx:
            break

    # Convert token indices back to words (excluding <sos>)
    tgt_words = [
        tgt_itos[idx] for idx in tgt_tokens[1:] if idx != eos_idx
    ]  # Exclude <sos> and <eos>
    return " ".join(tgt_words)


# --- 13. 测试翻译 ---
print("\n--- Testing Translation ---")
test_sentences = [
    "0",
    "1",
    "2",
    "3",
    "4",
]  # Add "two" which might be harder if similar to "ten"

for sentence in test_sentences:
    translation = translate(
        model, sentence, SRC_VOCAB, TGT_VOCAB, SRC_ITOS, TGT_ITOS, DEVICE
    )
    # Find the ground truth
    ground_truth = "Not in training data"
    for s, t in raw_data:
        if s == sentence:
            ground_truth = t
            break
    print(f"Translate '{sentence}' -> '{translation}' (Ground Truth: '{ground_truth}')")
