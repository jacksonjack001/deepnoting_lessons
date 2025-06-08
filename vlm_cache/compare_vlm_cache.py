import os
import time
from openai import OpenAI
import sys
import csv
from datetime import datetime

sys.path.insert(-1, "/data_ext/gradio_model_mix/")
from openai_tools import client

# 使用一个稳定、公开可访问的图片URL
# 这里我们使用梵高的《星夜》
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh\
    _-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-\
        _Google_Art_Project.jpg"


# 两个关于图片的不同问题
QUESTION_1 = "看下这张图什么风格，真实/虚幻，注意只返回2个字"
QUESTION_2 = "提取图片中的关键信息，控制在30个字以内"
QUESTION_2 = "画面的上半部分描绘了什么？"


def save_to_csv(results, filename="vision_model_benchmark3.csv"):
    """保存实验结果到CSV文件"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file_exists = os.path.exists(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # 写入表头
            writer.writerow(
                [
                    "时间戳",
                    "模型名称",
                    "首次请求时间(s)",
                    "缓存请求时间(s)",
                    "对照组时间(s)",
                    "加速比T1/T2",
                    "加速比T3/T2",
                    "是否成功",
                ]
            )

        # 计算加速比
        speedup_t1_t2 = time_1 / time_2 if time_2 > 0 else float("inf")
        speedup_t3_t2 = time_3 / time_2 if time_2 > 0 else float("inf")
        success = "是" if (time_2 < time_1 and time_2 < time_3) else "否"

        # 写入数据
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name,
                f"{time_1:.2f}",
                f"{time_2:.2f}",
                f"{time_3:.2f}",
                f"{speedup_t1_t2:.2f}",
                f"{speedup_t3_t2:.2f}",
                success,
            ]
        )


# 用于存储对话历史
for model_name in [
    "qwen-vl-max",
    "gpt-4.1-nano",
    "gpt-4.1",
    "doubao-1.5-vision-lite",
    "gemini-2.5-flash-preview",
    "claude-sonnet-4-20250514",
]:
    print(f"当前模型: {model_name}")
    conversation_history = []

    print("=" * 50)
    print("🚀 实验开始：验证视觉大模型的KV Cache机制")
    print(f"🖼️ 使用图片: 梵高《星夜》，使用模型：{model_name}")
    print("=" * 50 + "\n")

    # --- 实验一: 预热/缓存填充 ---
    print("--- 实验一: 首次请求 (包含图片URL) ---")
    print(f"问题: {QUESTION_1}")

    # 构建第一次请求的消息体
    messages_exp1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION_1},
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_URL},
                },
            ],
        }
    ]

    start_time_1 = time.time()
    response_1 = client.chat.completions.create(
        model=model_name, messages=messages_exp1, max_tokens=300, temperature=0
    )
    end_time_1 = time.time()
    time_1 = end_time_1 - start_time_1

    print(f"\n✅ 实验一完成，耗时: {time_1:.2f} 秒")
    response_content_1 = response_1.choices[0].message.content
    print(f"🤖 GPT-4o 回答: {response_content_1}")

    # 更新对话历史以用于实验二
    conversation_history.append(messages_exp1[0])
    conversation_history.append({"role": "assistant", "content": response_content_1})

    print("\n" + "=" * 50 + "\n")

    # --- 实验二: 测试缓存 ---
    print("--- 实验二: 第二轮请求 (利用缓存, 不再提供URL) ---")
    print(f"问题: {QUESTION_2}")

    # 构建第二次请求的消息体，只添加新问题
    messages_exp2 = conversation_history + [
        {
            "role": "user",
            "content": [{"type": "text", "text": QUESTION_2}],
        }
    ]

    start_time_2 = time.time()
    response_2 = client.chat.completions.create(
        model=model_name, messages=messages_exp2, max_tokens=300, temperature=0
    )
    end_time_2 = time.time()
    time_2 = end_time_2 - start_time_2

    print(f"\n✅ 实验二完成，耗时: {time_2:.2f} 秒")
    response_content_2 = response_2.choices[0].message.content
    print(f"🤖 GPT-4o 回答: {response_content_2}")
    print("\n" + "=" * 50 + "\n")

    # --- 实验三: 对照组 (无缓存) ---
    print("--- 实验三: 对照组 (全新会话, 重新提供URL) ---")
    print(f"问题: {QUESTION_2}")

    # 构建一个全新的请求，内容与实验二的问题相同，但需要重新提供URL
    messages_exp3 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION_2},
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_URL},
                },
            ],
        }
    ]

    start_time_3 = time.time()
    response_3 = client.chat.completions.create(
        model=model_name, messages=messages_exp3, max_tokens=300, temperature=0
    )
    end_time_3 = time.time()
    time_3 = end_time_3 - start_time_3

    print(f"\n✅ 实验三完成，耗时: {time_3:.2f} 秒")
    response_content_3 = response_3.choices[0].message.content
    print(f"🤖 GPT-4o 回答: {response_content_3}")

    print("\n" + "=" * 50)
    print("📊 实验结果分析:")
    print("=" * 50)
    print(f"⏱️ 首次请求时间 (T1): {time_1:.2f} 秒 (包含图片下载、编码和生成)")
    print(f"⏱️ 利用缓存时间 (T2): {time_2:.2f} 秒 (仅生成新文本)")
    print(f"⏱️ 对照组时间 (T3): {time_3:.2f} 秒 (重新下载、编码和生成)")
    print("\n结论:")
    if time_2 < time_1 and time_2 < time_3:
        print(
            f"✅ 验证成功! T2 ({time_2:.2f}s) 远小于 T1 ({time_1:.2f}s) 和 T3 ({time_3:.2f}s)。"
        )
        print(
            "这表明模型成功缓存了图片的视觉特征，在第二轮对话中无需重新处理图片，从而大幅提高了响应速度。"
        )
    else:
        print(
            "❌ 实验结果不符合预期。可能的原因包括网络波动、API服务器负载变化等。建议多运行几次以获得更稳定的结果。"
        )
    print("=" * 50)

    # 保存结果到CSV
    save_to_csv(
        {"model_name": model_name, "time_1": time_1, "time_2": time_2, "time_3": time_3}
    )

    print("📝 结果已保存到CSV文件")
    print("=" * 50)
