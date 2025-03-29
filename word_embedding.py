import numpy as np
import jieba
import csv
from gensim.models import Word2Vec, FastText
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 准备测试句子
apple_fruit_sentences = [
    "我喜欢吃新鲜的苹果",
    "这个苹果很甜很脆",
    "超市里的苹果打折了",
    "苹果富含维生素和纤维素",
]

apple_company_sentences = [
    "我买了一个苹果手机",
    "苹果公司发布了新产品",
    "苹果的市值突破万亿美元",
    "我在苹果商店购买了配件",
]

# 用于测试相似度的句子
test_sentences = [
    "苹果香蕉各有所爱",  # 水果相关
    "这个苹果真好吃",  # 水果相关
    "苹果的股票会上涨么",  # 科技公司相关
    "苹果的应用太封闭",  # 科技公司相关
]

# 所有句子合并，用于训练Word2Vec和FastText
all_sentences = apple_fruit_sentences + apple_company_sentences + test_sentences

# 使用jieba进行分词
tokenized_sentences = [list(jieba.cut(sent)) for sent in all_sentences]

# 1. 训练Word2Vec模型
word2vec_model = Word2Vec(
    sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4
)

# 2. 训练FastText模型
fasttext_model = FastText(
    sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4
)

# 3. 加载预训练的BERT模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertModel.from_pretrained("bert-base-chinese")
bert_model.eval()


# 获取Word2Vec的句子向量（词向量平均）
def get_word2vec_sentence_embedding(sentence):
    words = list(jieba.cut(sentence))
    word_vectors = []
    for word in words:
        if word in word2vec_model.wv:
            word_vectors.append(word2vec_model.wv[word])
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(100)


# 获取FastText的句子向量（词向量平均）
def get_fasttext_sentence_embedding(sentence):
    words = list(jieba.cut(sentence))
    word_vectors = []
    for word in words:
        word_vectors.append(fasttext_model.wv[word])
    return np.mean(word_vectors, axis=0)


# 获取BERT的句子向量 - 修复维度问题
def get_bert_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # 使用[CLS]标记的嵌入作为句子表示，并确保返回一个1维数组
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()


# 计算所有句子的嵌入
word2vec_embeddings = {}
fasttext_embeddings = {}
bert_embeddings = {}

# 计算所有句子的嵌入
for sentence in all_sentences:
    word2vec_embeddings[sentence] = get_word2vec_sentence_embedding(sentence)
    fasttext_embeddings[sentence] = get_fasttext_sentence_embedding(sentence)
    bert_embeddings[sentence] = get_bert_sentence_embedding(sentence)

# 创建CSV文件保存相似度结果
with open("sentence_similarities.csv", "w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)

    # 写入标题行
    csv_writer.writerow(["模型", "句子类型", "测试句子", "相似度"])

    # 计算相似度并保存结果
    def calculate_and_save_similarities(model_name, embeddings):
        print(f"\n{model_name} 模型句子相似度结果:")

        # 计算水果义"苹果"句子与测试句子的平均相似度
        print("\n水果义'苹果'句子与测试句子的相似度:")
        for test_sent in test_sentences:
            avg_sim = 0
            for apple_sent in apple_fruit_sentences:
                sim = cosine_similarity(
                    [embeddings[apple_sent]], [embeddings[test_sent]]
                )[0][0]
                avg_sim += sim
            avg_sim /= len(apple_fruit_sentences)
            print(f"'{test_sent}': {avg_sim:.4f}")
            csv_writer.writerow([model_name, "水果义苹果", test_sent, f"{avg_sim:.4f}"])

        # 计算公司义"苹果"句子与测试句子的平均相似度
        print("\n公司义'苹果'句子与测试句子的相似度:")
        for test_sent in test_sentences:
            avg_sim = 0
            for apple_sent in apple_company_sentences:
                sim = cosine_similarity(
                    [embeddings[apple_sent]], [embeddings[test_sent]]
                )[0][0]
                avg_sim += sim
            avg_sim /= len(apple_company_sentences)
            print(f"'{test_sent}': {avg_sim:.4f}")
            csv_writer.writerow([model_name, "公司义苹果", test_sent, f"{avg_sim:.4f}"])

    # 展示各模型的相似度结果
    calculate_and_save_similarities("Word2Vec", word2vec_embeddings)
    calculate_and_save_similarities("FastText", fasttext_embeddings)
    calculate_and_save_similarities("BERT", bert_embeddings)

# 创建CSV文件保存二义性区分能力结果
with open("disambiguation_ability.csv", "w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)

    # 写入标题行
    csv_writer.writerow(
        [
            "模型",
            "水果义与水果相关相似度",
            "水果义与公司相关相似度",
            "公司义与水果相关相似度",
            "公司义与公司相关相似度",
            "二义性区分能力指标",
        ]
    )

    # 进一步分析：比较不同模型对二义性的区分能力
    print("\n\n各模型对'苹果'二义性的区分能力比较:")

    models = {
        "Word2Vec": word2vec_embeddings,
        "FastText": fasttext_embeddings,
        "BERT": bert_embeddings,
    }

    for model_name, embeddings in models.items():
        print(f"\n{model_name}模型:")

        # 计算水果义与水果相关测试句子的平均相似度
        fruit_to_fruit_sim = 0
        for apple_sent in apple_fruit_sentences:
            for test_sent in test_sentences[:2]:  # 前两个是水果相关
                sim = cosine_similarity(
                    [embeddings[apple_sent]], [embeddings[test_sent]]
                )[0][0]
                fruit_to_fruit_sim += sim
        fruit_to_fruit_sim /= len(apple_fruit_sentences) * 2

        # 计算水果义与公司相关测试句子的平均相似度
        fruit_to_company_sim = 0
        for apple_sent in apple_fruit_sentences:
            for test_sent in test_sentences[2:]:  # 后两个是公司相关
                sim = cosine_similarity(
                    [embeddings[apple_sent]], [embeddings[test_sent]]
                )[0][0]
                fruit_to_company_sim += sim
        fruit_to_company_sim /= len(apple_fruit_sentences) * 2

        # 计算公司义与水果相关测试句子的平均相似度
        company_to_fruit_sim = 0
        for apple_sent in apple_company_sentences:
            for test_sent in test_sentences[:2]:  # 前两个是水果相关
                sim = cosine_similarity(
                    [embeddings[apple_sent]], [embeddings[test_sent]]
                )[0][0]
                company_to_fruit_sim += sim
        company_to_fruit_sim /= len(apple_company_sentences) * 2

        # 计算公司义与公司相关测试句子的平均相似度
        company_to_company_sim = 0
        for apple_sent in apple_company_sentences:
            for test_sent in test_sentences[2:]:  # 后两个是公司相关
                sim = cosine_similarity(
                    [embeddings[apple_sent]], [embeddings[test_sent]]
                )[0][0]
                company_to_company_sim += sim
        company_to_company_sim /= len(apple_company_sentences) * 2

        print(f"水果义'苹果'与水果相关句子的平均相似度: {fruit_to_fruit_sim:.4f}")
        print(f"水果义'苹果'与公司相关句子的平均相似度: {fruit_to_company_sim:.4f}")
        print(f"公司义'苹果'与水果相关句子的平均相似度: {company_to_fruit_sim:.4f}")
        print(f"公司义'苹果'与公司相关句子的平均相似度: {company_to_company_sim:.4f}")

        # 计算区分能力指标（相同语义相似度与不同语义相似度之差的平均）
        disambiguation_score = (
            (fruit_to_fruit_sim - fruit_to_company_sim)
            + (company_to_company_sim - company_to_fruit_sim)
        ) / 2
        print(f"二义性区分能力指标: {disambiguation_score:.4f}")

        # 将结果写入CSV
        csv_writer.writerow(
            [
                model_name,
                f"{fruit_to_fruit_sim:.4f}",
                f"{fruit_to_company_sim:.4f}",
                f"{company_to_fruit_sim:.4f}",
                f"{company_to_company_sim:.4f}",
                f"{disambiguation_score:.4f}",
            ]
        )

print(
    "\n结果已保存到 'sentence_similarities.csv' 和 'disambiguation_ability.csv' 文件中"
)
