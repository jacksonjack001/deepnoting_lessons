import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# 超参数
LR = 3e-5
GAMMA = 0.99
CLIP_EPSILON = 0.2
KL_PENALTY = 0.01
BATCH_SIZE = 4
EPOCHS = 50

# 初始化模型和分词器
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 测试数据
test_prompts = [
    "The weather is nice today,",
    "I want to learn about",
    "Artificial intelligence is",
    "The best programming language is",
]


def calculate_reward(response):
    """简单的奖励函数：鼓励生成长文本"""
    return len(response.split()) / 20  # 标准化到0-1范围


def generate_response(prompt, max_length=30):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def ppo_train_step(prompts):
    # 生成初始响应
    old_responses = [generate_response(p) for p in prompts]
    old_rewards = torch.tensor(
        [calculate_reward(r) for r in old_responses], dtype=torch.float32
    )

    # 存储旧策略的概率
    old_probs = []
    for prompt, response in zip(prompts, old_responses):
        inputs = tokenizer(prompt + response, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        old_probs.append(torch.exp(-outputs.loss))  # 近似概率

    # 训练多个epoch
    for _ in range(3):
        # 生成新响应
        new_responses = [generate_response(p) for p in prompts]
        new_rewards = torch.tensor(
            [calculate_reward(r) for r in new_responses], dtype=torch.float32
        )

        # 计算新策略的概率
        new_probs = []
        for prompt, response in zip(prompts, new_responses):
            inputs = tokenizer(prompt + response, return_tensors="pt", truncation=True)
            outputs = model(**inputs, labels=inputs["input_ids"])
            new_probs.append(torch.exp(-outputs.loss))

        # 计算比率和损失
        ratios = torch.stack([n / o for n, o in zip(new_probs, old_probs)])
        advantages = new_rewards - old_rewards

        # PPO损失计算
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        loss = -torch.min(surr1, surr2).mean() + KL_PENALTY * (ratios.log().mean()) ** 2

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return new_rewards.mean().item()


# 训练循环
for epoch in range(EPOCHS):
    # 随机选择训练提示
    batch_prompts = np.random.choice(test_prompts, size=BATCH_SIZE)

    # 执行PPO训练步骤
    avg_reward = ppo_train_step(batch_prompts)

    # 测试当前模型
    print(f"\nEpoch {epoch+1}/{EPOCHS}, Avg Reward: {avg_reward:.3f}")
    test_response = generate_response(test_prompts[0])
    print(f"Example response: {test_response}")
