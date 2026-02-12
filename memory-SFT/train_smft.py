import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
from typing import Dict, List, Optional
import torch
import numpy as np
import random
import os
from collections import deque
from pathlib import Path
import csv

def set_seed(seed=42):
    """设置所有随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    
    # 确保CUDA的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在代码开始处调用
set_seed(42)

# ================= 配置部分 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"  # 指定显卡
MODEL_PATH = "/data_ext1/models/Qwen2.5-7B-Instruct"
DATA_PATH = "/data_ext1/data/Chinese-SimpleQA"
OUTPUT_DIR = "./output_sparse_memory_qwen"

# 超参数
MEMORY_LAYERS = [14]  # 选择替换第12层FFN为Memory Layer
NUM_KEYS = 32 ** 2    # Product Keys总数
TOP_K = 32            # 检索 Top K
DIM_HEAD = 128        # 记忆投影维度
UPDATE_TOP_T = 32     # 稀疏微调 Top T

# ================= 核心模块：Product Key Memory =================

class ProductKeyMemory(nn.Module):
    """
    实现论文描述的 Memory Layer，包含 Product Keys 检索机制。
    替代原本的 FFN (MLP)。
    """
    def __init__(self, input_dim, output_dim, num_keys, top_k=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_keys = num_keys
        self.top_k = top_k
        
        self.sub_key_size = int(math.sqrt(num_keys))
        assert self.sub_key_size ** 2 == num_keys, "num_keys must be a perfect square"
        
        # 【修复 1】：使用更小的初始化，防止数值爆炸
        self.query_proj = nn.Linear(input_dim, input_dim)
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.01)
        nn.init.zeros_(self.query_proj.bias)
        
        self.half_dim = input_dim // 2
        
        # 【修复 2】：keys 使用更保守的初始化
        self.keys1 = nn.Parameter(torch.randn(self.sub_key_size, self.half_dim) * 0.01)
        self.keys2 = nn.Parameter(torch.randn(self.sub_key_size, self.half_dim) * 0.01)
        
        # 【修复 3】：values 初始化为接近0，避免初期输出过大
        self.values = nn.Embedding(num_keys, output_dim)
        nn.init.normal_(self.values.weight, mean=0.0, std=0.01)
        
        # 【修复 4】：output_gate 初始化偏向于保留原始输入
        self.output_gate = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.output_gate.weight, gain=0.01)
        nn.init.constant_(self.output_gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12，初期主要保留原输入
        
        self.register_buffer("background_counts", torch.ones(num_keys)) 
        self.register_buffer("total_steps", torch.tensor(0))
        
        self.current_batch_indices = None
        self.current_batch_counts = None
        
        # 【修复 5】：添加 LayerNorm 稳定训练
        self.layer_norm = nn.LayerNorm(output_dim)

    def _get_indices(self, query):
        bs, seq_len, dim = query.size()
        q1 = query[:, :, :self.half_dim] 
        q2 = query[:, :, self.half_dim:] 
        
        # 【修复 6】：对 query 进行 L2 归一化，防止分数过大
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)
        
        # 【修复 7】：对 keys 也进行归一化
        k1 = F.normalize(self.keys1, p=2, dim=-1)
        k2 = F.normalize(self.keys2, p=2, dim=-1)
        
        scores1 = torch.matmul(q1, k1.t()) 
        scores2 = torch.matmul(q2, k2.t()) 
        
        # 【修复 8】：添加温度系数，控制分数范围
        temperature = 0.1
        combined_scores = (scores1.unsqueeze(-1) + scores2.unsqueeze(-2)) / temperature
        combined_scores = combined_scores.view(bs, seq_len, -1)
        
        # 【修复 9】：限制分数范围，防止 softmax 溢出
        combined_scores = torch.clamp(combined_scores, min=-10, max=10)
        
        topk_scores, topk_indices = torch.topk(combined_scores, k=self.top_k, dim=-1)
        return topk_scores, topk_indices

    def forward(self, x):
        # 【修复 10】：检查输入是否包含 nan 或 inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in input to ProductKeyMemory")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        q = self.query_proj(x)
        scores, indices = self._get_indices(q)
        
        if self.training:
            flat_indices = indices.flatten()
            self.current_batch_indices = flat_indices
            counts = torch.bincount(flat_indices, minlength=self.num_keys)
            self.current_batch_counts = counts
            
            alpha = 0.01
            self.background_counts = (1 - alpha) * self.background_counts + alpha * counts.float().detach()

        memory_values = self.values(indices)
        
        # 【修复 11】：使用稳定的 softmax
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).unsqueeze(-1)
        attn_weights = attn_weights.to(memory_values.dtype)
        
        weighted_output = torch.sum(memory_values * attn_weights, dim=2)
        
        # 【修复 12】：限制 gate 输出范围
        gate_logits = self.output_gate(x)
        gate_logits = torch.clamp(gate_logits, min=-5, max=5)
        gate = torch.sigmoid(gate_logits)
        
        # 【修复 13】：残差连接 + LayerNorm
        final_output = gate * weighted_output + (1 - gate) * x
        final_output = self.layer_norm(final_output)
        
        # 【修复 14】：检查输出
        if torch.isnan(final_output).any() or torch.isinf(final_output).any():
            print("Warning: NaN or Inf detected in output of ProductKeyMemory")
            final_output = torch.nan_to_num(final_output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return final_output

# ================= 稀疏微调 Hook 逻辑 =================

def get_gradient_mask_hook(memory_layer: ProductKeyMemory, top_t: int):
    def hook(grad):
        # 【修复 15】：检查梯度
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print("Warning: NaN or Inf detected in gradient")
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if memory_layer.current_batch_counts is None:
            return grad

        device = grad.device
        tf = memory_layer.current_batch_counts.float().to(device)
        total_steps = memory_layer.total_steps.float().to(device)
        bg_counts = memory_layer.background_counts.to(device)
        
        # 【修复 16】：更稳定的 IDF 计算
        idf = torch.log(total_steps + 10.0) - torch.log(bg_counts + 1.0)
        idf = torch.clamp(idf, min=0, max=10)
        
        scores = tf * idf
        
        active_mask = tf > 0
        scores[~active_mask] = -float('inf')
        
        k_actual = min(top_t, active_mask.sum().item())
        if k_actual == 0:
            return grad * 0.0
            
        _, top_indices = torch.topk(scores, k=k_actual)
        
        mask = torch.zeros_like(grad)
        mask[top_indices] = 1.0
        
        return grad * mask
        
    return hook

# ================= 数据处理 =================

def prepare_data(tokenizer, data_path):
    try:
        dataset = load_dataset("json", data_files=os.path.join(data_path, "chinese_simpleqa.jsonl"))
    except:
        dataset = load_dataset(data_path)

    def preprocess(examples):
        inputs = [f"用户: {q}\nAI: {a}" for q, a in zip(examples['question'], examples['answer'])]
        print(inputs[:5])
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        labels = model_inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_ds = dataset['train'].map(preprocess, batched=True, remove_columns=dataset['train'].column_names)
    tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return tokenized_ds

# ================= 训练曲线保存 =================

def save_loss_artifacts(loss_history: List[float], output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "loss_curve.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for step, loss in enumerate(loss_history, start=1):
            writer.writerow([step, f"{loss:.8f}"])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(1, len(loss_history) + 1))
        plt.figure(figsize=(10, 4))
        plt.plot(steps, loss_history, label="loss", linewidth=1.0)

        window = 10
        if len(loss_history) >= window:
            moving_avg = []
            running = 0.0
            for i, loss in enumerate(loss_history):
                running += loss
                if i >= window:
                    running -= loss_history[i - window]
                if i >= window - 1:
                    moving_avg.append(running / window)
                else:
                    moving_avg.append(None)

            ma_steps = [s for s, v in zip(steps, moving_avg) if v is not None]
            ma_vals = [v for v in moving_avg if v is not None]
            plt.plot(ma_steps, ma_vals, label=f"moving_avg_{window}", linewidth=1.5)

        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        png_path = out_dir / "loss_curve.png"
        plt.savefig(png_path, dpi=160)
        plt.close()
        print(f"Saved loss curve: {png_path} (CSV: {csv_path})")
    except Exception as e:
        print(f"Saved loss CSV: {csv_path} (plot skipped: {e})")

# ================= 主流程 =================

def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 【修复 17】：使用 bfloat16 代替 float16，更稳定
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # 从 float16 改为 bfloat16
    )
    
    print("Performing model surgery...")
    for param in model.parameters():
        param.requires_grad = False
        
    target_hidden_size = model.config.hidden_size
    layers = model.model.layers
    memory_modules = []
    
    for layer_idx in MEMORY_LAYERS:
        print(f"Replacing Layer {layer_idx} MLP with ProductKeyMemory...")
        
        original_mlp = layers[layer_idx].mlp
        target_device = original_mlp.down_proj.weight.device
        
        mem_layer = ProductKeyMemory(
            input_dim=target_hidden_size,
            output_dim=target_hidden_size,
            num_keys=NUM_KEYS,
            top_k=TOP_K
        ).to(target_device).to(torch.bfloat16)  # 改为 bfloat16
        
        layers[layer_idx].mlp = mem_layer
        memory_modules.append(mem_layer)
        
        for param in mem_layer.parameters():
            param.requires_grad = True
            
        mem_layer.values.weight.register_hook(get_gradient_mask_hook(mem_layer, UPDATE_TOP_T))

    print("Model surgery complete.")
    
    print("Processing data...")
    train_dataset = prepare_data(tokenizer, DATA_PATH)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)  # 【修复 18】：减小 batch size
    
    # 【修复 19】：降低学习率，添加梯度裁剪
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=5e-4,  # 从 1e-4 降低到 5e-5
        weight_decay=0.01
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    model.train()
    print("Starting Sparse Memory Finetuning...")
    
    global_step = 0
    total_steps = len(train_loader) * 3
    loss_history: List[float] = []
    recent_losses = deque(maxlen=10)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(6):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 【修复 20】：检查 loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at step {global_step}, skipping...")
                continue
            
            loss.backward()
            
            # 【修复 21】：梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            for mem in memory_modules:
                mem.total_steps += 1
            
            global_step += 1
            loss_value = float(loss.detach().cpu().item())
            loss_history.append(loss_value)
            recent_losses.append(loss_value)

            if global_step % 10 == 0:
                print(f"Epoch {epoch}, Step {global_step}/{total_steps}, Loss: {loss.item():.4f}")
                
            if global_step % 1000 == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    save_loss_artifacts(loss_history, OUTPUT_DIR)

    if len(recent_losses) == 0:
        print("No valid optimization steps recorded; cannot compute last-10-step average loss.")
    else:
        last_k = min(10, len(recent_losses))
        last_avg = sum(list(recent_losses)[-last_k:]) / last_k
        print(f"Last {last_k} steps average loss: {last_avg:.6f}")

    print(f"Training finished. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
