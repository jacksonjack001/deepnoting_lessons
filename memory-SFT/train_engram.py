import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
from sympy import isprime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from tokenizers import normalizers, Regex
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import matplotlib.pyplot as plt
import os
from transformers import TrainerCallback

# ==============================================================================
# [Part 1: Engram Architecture Implementation]
# (Modified slightly to integrate with global configs dynamically)
# ==============================================================================

class LossPlottingCallback(TrainerCallback):
    """
    自定义 Callback，用于在训练结束时绘制 Loss 曲线
    """
    def on_train_end(self, args, state, control, **kwargs):
        # 仅在主进程（Rank 0）进行绘图，避免多进程冲突
        if args.local_rank not in [-1, 0]:
            return

        # 从 log_history 中提取数据
        train_steps = []
        train_losses = []
        eval_steps = []
        eval_losses = []

        for log in state.log_history:
            if "loss" in log:
                train_steps.append(log["step"])
                train_losses.append(log["loss"])
            if "eval_loss" in log:
                eval_steps.append(log["step"])
                eval_losses.append(log["eval_loss"])

        # 开始绘图
        plt.figure(figsize=(10, 6))
        
        if train_losses:
            plt.plot(train_steps, train_losses, label="Training Loss", color='blue')
        
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label="Validation Loss", color='red', linestyle='--')

        plt.xlabel("Global Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.grid(True)

        # 保存图片到 output_dir
        plot_path = os.path.join(args.output_dir, "loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\n[INFO] Loss curve saved to: {plot_path}")


@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "" # Will be updated in main
    engram_vocab_size: List[int] = field(default_factory=lambda: [100000, 100000]) # Adjusted for demo
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    # Select layers to inject Engram. E.g., layer 1 and 15
    layer_ids: List[int] = field(default_factory=lambda: [1, 15]) 
    pad_id: int = 0 # Will be updated based on tokenizer
    seed: int = 42
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024 # Will be updated from model config
    hc_mult: int = 1        # Set to 1 for standard Transformer integration
    vocab_size: int = 129280
    num_layers: int = 30

# Global configs instances
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

class CompressedTokenizer:
    def __init__(self, tokenizer_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []
        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        # Handle torch tensors or numpy arrays
        if isinstance(input_ids, torch.Tensor):
            arr = input_ids.cpu().numpy().astype(np.int64)
        else:
            arr = np.asarray(input_ids, dtype=np.int64)
            
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        # Safety clip for vocab size mismatch
        valid_ids = np.clip(valid_ids, 0, len(self.lookup_table) - 1)
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)

class ShortConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 4, dilation: int = 1, norm_eps: float = 1e-5, hc_mult: int = 1, activation: bool = True):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, L, G, C)
        B, T, G, C = x.shape
        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        x_norm = torch.cat(normed_chunks, dim=-1) # (B, L, G*C)
        x_bct = x_norm.transpose(1, 2) # (B, G*C, L)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]
        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        return y

def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(self, engram_vocab_size, max_ngram_size, n_embed_per_ngram, n_head_per_ngram, layer_ids, tokenizer_name_or_path, pad_id, seed):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids
        self.compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path=tokenizer_name_or_path)            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            # Safe lookup for pad_id
            if self.pad_id < len(self.compressed_tokenizer.lookup_table):
                self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])
            else:
                self.pad_id = 0

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers
        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                # Handle case where engram_vocab_size list is shorter than max_ngram_size
                idx = min(ngram - 2, len(self.vocab_size_per_ngram) - 1)
                vocab_size = self.vocab_size_per_ngram[idx]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                for _ in range(num_head):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]
        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)), mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        return output

class Engram(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
    
    def forward(self, hidden_states, input_ids):
        """
        hidden_states: [B, L, HC_MULT, D] (We will adapt [B, L, D] to this inside wrapper)
        input_ids: [B, L]
        """
        # Hash calculation is on CPU usually due to numpy, move to device
        device = hidden_states.device
        
        # Optimization: Cache hashes if possible, but for training we compute on fly
        # Note: hash_mapping.hash returns numpy, need to convert to tensor on correct device
        hashes_numpy = self.hash_mapping.hash(input_ids.cpu().numpy())[self.layer_id]
        hash_input_ids = torch.from_numpy(hashes_numpy).to(device)
        
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        
        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        
        gates = torch.stack(gates, dim=2)
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)
        return output 

# ==============================================================================
# [Part 2: Integration Logic - Wrappers & Hooks]
# ==============================================================================

# Context to store input_ids during forward pass
class GlobalInputContext:
    _input_ids = None
    
    @classmethod
    def set_input_ids(cls, input_ids):
        cls._input_ids = input_ids
        
    @classmethod
    def get_input_ids(cls):
        return cls._input_ids

class EngramLayerWrapper(nn.Module):
    """
    Wraps a standard Transformer layer (e.g., Qwen2DecoderLayer) to inject Engram.
    """
    def __init__(self, original_layer, engram_module):
        super().__init__()
        self.original_layer = original_layer
        self.engram = engram_module
        
    def forward(self, hidden_states, *args, **kwargs):
        # 1. Retrieve input_ids from global context
        input_ids = GlobalInputContext.get_input_ids()
        
        if input_ids is not None:
            # 2. Adapt dimensions: Standard [B, L, D] -> Engram [B, L, 1, D]
            # Assuming hc_mult=1 for standard integration
            hidden_states_expanded = hidden_states.unsqueeze(2) 
            
            # 3. Compute Engram features
            # Note: Engram returns the *residual* component
            engram_out = self.engram(hidden_states_expanded, input_ids)
            
            # 4. Add residual to hidden states
            hidden_states = hidden_states + engram_out.squeeze(2)
            
        # 5. Pass through original layer
        return self.original_layer(hidden_states, *args, **kwargs)

def input_capture_hook(module, args, kwargs):
    """
    Hook to be registered on the model's forward pass to capture input_ids.
    """
    input_ids = kwargs.get('input_ids', None)
    if input_ids is None and len(args) > 0:
        input_ids = args[0]
    
    if input_ids is not None:
        GlobalInputContext.set_input_ids(input_ids)

# ==============================================================================
# [Part 3: Training Script]
# ==============================================================================

# Environment Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Configuration
MODEL_PATH = '/data/lz_data/models/Qwen3-8B/' # Replace with your actual path
DATA_PATH = "/data/lz_data/aicr/data/fxm/doc_train_utf8.jsonl"
VAL_PATH = "/data/lz_data/aicr/data/fxm/doc_val_utf8.jsonl"
OUTPUT_DIR = "/data/lz_data/aicr/output/fxm_engram_sft"

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 64
LEARNING_RATE = 1e-4
EPOCHS = 3
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Update Global Configs based on Model/Tokenizer
    engram_cfg.tokenizer_name_or_path = MODEL_PATH
    engram_cfg.pad_id = tokenizer.pad_token_id
    
    # 3. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )
    model.enable_input_require_grads()
    
    # 4. Sync Backbone Config
    backbone_config.hidden_size = model.config.hidden_size
    backbone_config.vocab_size = model.config.vocab_size
    backbone_config.num_layers = model.config.num_hidden_layers
    backbone_config.hc_mult = 1 # Force 1 for standard Transformer integration

    print("Injecting Engram modules...")
    # 5. Inject Engram Modules
    # We iterate over layers and replace specific ones with the wrapper
    # Qwen2 structure: model.model.layers
    layers = model.model.layers
    
    for layer_id in engram_cfg.layer_ids:
        if 0 <= layer_id < len(layers):
            print(f"  -> Injecting Engram at layer {layer_id}")
            original_layer = layers[layer_id]
            
            # Initialize Engram module for this layer
            # Note: Engram init uses global configs we just updated
            engram_module = Engram(layer_id=layer_id).to(model.device).to(torch.bfloat16)
            
            # Wrap
            wrapped_layer = EngramLayerWrapper(original_layer, engram_module)
            layers[layer_id] = wrapped_layer
        else:
            print(f"  [Warning] Layer ID {layer_id} out of bounds.")

    # 6. Register Hook to capture input_ids
    # We hook onto the main model body (model.model)
    model.model.register_forward_pre_hook(input_capture_hook, with_kwargs=True)


    # 7. Apply LoRA
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    
    # 8. Set Trainable Parameters
    # LoRA sets base model to requires_grad=False and adapters to True.
    # We need to ensure Engram parameters are ALSO True.
    trainable_params = 0
    all_param = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        # Enable gradients for Engram modules
        if "engram" in name:
            param.requires_grad = True
        
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")

    # ================= Data Processing (Same as your code) =================
    print(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files={"train": DATA_PATH, "validation": VAL_PATH})

    def process_func(example):
        messages = example.get('messages', [])
        if not messages: return {'input_ids': [], 'labels': [], 'attention_mask': []}
        
        last_assistant_idx = -1
        for i, msg in enumerate(messages):
            if msg.get('role') == 'assistant': last_assistant_idx = i
        if last_assistant_idx == -1: return {'input_ids': [], 'labels': [], 'attention_mask': []}
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer(text, max_length=MAX_LENGTH, padding=False, truncation=True)
        
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = list(input_ids)

        prompt_messages = messages[:last_assistant_idx]
        if prompt_messages:
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, max_length=MAX_LENGTH, truncation=True)["input_ids"]
            prompt_len = len(prompt_ids)
            if prompt_len < len(labels):
                labels[:prompt_len] = [-100] * prompt_len
            else:
                labels = [-100] * len(labels)
        
        if input_ids and input_ids[-1] != tokenizer.eos_token_id:
            input_ids.append(tokenizer.eos_token_id)
            attention_mask.append(1)
            labels.append(tokenizer.eos_token_id)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_dataset = dataset["train"].map(process_func, remove_columns=dataset["train"].column_names).filter(lambda x: len(x['input_ids']) > 0)
    eval_dataset = dataset["validation"].map(process_func, remove_columns=dataset["validation"].column_names).filter(lambda x: len(x['input_ids']) > 0)

    # ================= Training Arguments =================
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple): logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        mask = labels != -100
        correct = (preds == labels) & mask
        accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0
        return {"accuracy": float(accuracy)}

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        eval_accumulation_steps=1,
        per_device_eval_batch_size=1,
        logging_steps=10,
        num_train_epochs=EPOCHS,
        save_steps=10,
        learning_rate=LEARNING_RATE,
        save_on_each_node=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        bf16=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False, # Set True if you encounter errors with Engram params not being used
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        compute_metrics=compute_metrics,
        callbacks=[LossPlottingCallback()], 
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
