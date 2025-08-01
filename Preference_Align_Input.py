import os
import time
import re
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import clean_text
# Packages for Similarity Calculation
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import json

dataset = 'dbbook2014'
if dataset in ['dbbook2014', 'book-crossing', 'test_dataset']:
    field = 'books'
else:
    field = 'movies'

# Select in '' or '_r'. Two text cleaning strategy.
r = '_r'
C = 350
max_his_num = 30

# Select in [st, tfidf]; 'st' means 'SentenceTransformer'
cluster_type = 'tfidf'
data_path = f'./data/{dataset}'
# Replace your file name
output_path = f'batch_output/{dataset}_max{max_his_num}_output.jsonl'

# if dataset in ['dbbook2014', 'ml1m']:
#     item_list = pd.read_csv(data_path + '/item_list.txt', sep=' ')
#     entity_list = pd.read_csv(data_path + '/entity_list.txt', sep=' ')
# else:
#     item_list = list()
#     entity_list = list()
#     lines = open(data_path + '/item_list.txt', 'r').readlines()
#     for l in lines[1:]:
#         if l == "\n":
#             continue
#         tmps = l.replace("\n", "").strip()
#         elems = tmps.split(' ')
#         org_id = elems[0]
#         remap_id = elems[1]
#         if len(elems[2:]) == 0:
#             continue
#         title = ' '.join(elems[2:])
#         item_list.append([org_id, remap_id, title])
#     item_list = pd.DataFrame(item_list, columns=['org_id', 'remap_id', 'entity_name'])
#     lines = open(data_path + '/entity_list.txt', 'r').readlines()
#     for l in lines[1:]:
#         if l == "\n":
#             continue
#         tmps = l.replace("\n", "").strip()
#         elems = tmps.split()
#         org_id = elems[0]
#         remap_id = elems[1]
#         # entity_name = elems[2]
#         entity_list.append([org_id, remap_id])
#     entity_list = pd.DataFrame(entity_list, columns=['org_id', 'remap_id'])
#
# kg = pd.read_csv(data_path + '/kg_final.txt', sep=' ', names=['head', 'relation', 'tail'])
# relation = pd.read_csv(data_path + '/relation_list.txt', sep=' ')
# entity_list['remap_id'] = entity_list['remap_id'].astype(int)
#
# relation_intent = kg['relation'].max() + 1
# relation_sim = kg['relation'].max() + 2
# items = item_list['remap_id'].to_list()

# Avoid too short/long generated text (outliers)
lower = 1
upper = 50
llm_answer = []
intent_triples = []
sim_triples = set()
intents_list = []
check_intent = []
waste = []
with open(output_path, mode='r') as f:
    for answer in jsonlines.Reader(f):
        row_id = answer['custom_id']
        raw_intents = answer['response']['body']['choices'][0]['message']['content']
        clean_intents = [clean_text(it) for it in raw_intents.strip().split(',')]

        intents = []
        for it in clean_intents:
            if r == '':
                if len(it.split()) > lower and len(it.split()) <= upper:
                    intents.append(it)
                else:
                    print(f'waste:', it.split())
                    waste.append(it)
            elif r == '_r':
                if len(it) > lower and len(it) <= upper:
                    intents.append(it)
                else:
                    print(f'waste:', it)
                    waste.append(it)

        for it in intents:
            intent_triples.append([row_id, it])
        intents_list.extend(intents)


def encode_intents(cluster_type, intents_list):
    if cluster_type == 'st':
        # Sentence Transformer
        # Initialize the model
        device = 'cuda:0'
        # 'replace into your Sentence Transformer path'
        model = SentenceTransformer('./all-MiniLM-L6-v2')
        model.to(device)
        sentences = intents_list
        embeddings = model.encode(sentences, device=device)
    elif cluster_type == 'tfidf':
        # TFIDF
        sentences = intents_list
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer(max_df=0.8, ngram_range=(1, 2))
        embeddings = vectorizer.fit_transform(sentences)
    return embeddings


embeddings = encode_intents(cluster_type, intents_list)

print(f'cluster num: {C}')
kmeans = KMeans(n_clusters=C, n_init='auto')  # You can adjust the number of clusters as needed
cluster_ids = kmeans.fit_predict(embeddings)
# Create the output dictionary
output_dict = dict(zip(intents_list, cluster_ids))

llm = "gpt-3.5-turbo-0125"
version = 'v1'

# 保存 intent_triples
user_intent_file = f'User_Intent/{dataset}_user_intent_{llm}.json'
with open(user_intent_file, 'w', encoding='utf-8') as f:
    json.dump(intent_triples, f, ensure_ascii=False, indent=2)
print(f"intent_triples 已保存到: {user_intent_file}")

def convert_numpy_to_python(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 转换后保存
output_dict = convert_numpy_to_python(output_dict)
intent_cluster_file = f'User_Intent/{dataset}_intent_cluster_{llm}.json'
with open(intent_cluster_file, 'w', encoding='utf-8') as f:
    json.dump(output_dict, f, ensure_ascii=False, indent=2)
print(f"output_dict 已保存到: {intent_cluster_file}")

# def get_system_prompt():
#     """定义系统提示词"""
#     return """You are an expert in intent analysis and text processing. Your task is to merge similar intents within a given group.
#
# Rules:
# 1. Only merge intents that are semantically very similar or essentially express the same meaning
# 2. Keep intents that are clearly different, even if they are in the same domain
# 3. For merged intents, choose the most representative and clear expression
# 4. Output must be in strict JSON format
#
# Output format:
# {
#     "merged_intents": [
#         {
#             "representative_intent": "the most representative intent text",
#             "merged_from": ["intent1", "intent2", "intent3"]
#         }
#     ],
#     "unchanged_intents": ["intent4", "intent5"]
# }
#
# Important:
# - If no intents need merging, put all intents in "unchanged_intents"
# - Each intent must appear in exactly one place (either as merged or unchanged)
# - Use the exact original text for intent names"""

# def get_system_prompt():
#     """定义系统提示词"""
#     return """You are an expert in intent analysis and text processing. Your task is to aggressively merge similar intents within a given group into broader, coarse-grained categories.
#
# Rules:
# 1. PRIORITIZE MERGING - Always look for opportunities to combine intents into broader categories
# 2. Merge intents that share similar themes, domains, or purposes, even if they have different specific expressions
# 3. Create coarse-grained, high-level intent categories that encompass multiple related specific intents
# 4. For merged intents, choose the most general and comprehensive expression that covers all merged intents
# 5. Only keep intents separate if they belong to completely different domains or have fundamentally different purposes
# 6. When in doubt, merge rather than keep separate
# 7. Output must be in strict JSON format
#
# Merging Guidelines:
# - Merge intents with similar semantic meaning or purpose
# - Merge intents from the same domain or category (e.g., all shopping-related, all information-seeking, etc.)
# - Merge intents that represent different ways of expressing the same underlying need
# - Create broader umbrella categories that can encompass multiple specific use cases
# - Aim for fewer, more comprehensive intent categories rather than many specific ones
#
# Output format:
# {
#     "merged_intents": [
#         {
#             "representative_intent": "the most general and comprehensive intent description",
#             "merged_from": ["intent1", "intent2", "intent3", "intent4"]
#         }
#     ],
#     "unchanged_intents": ["only truly unique intents that cannot be merged"]
# }
#
# Important:
# - Favor aggressive merging - err on the side of combining rather than separating
# - Each intent must appear in exactly one place (either as merged or unchanged)
# - Use the most general and inclusive language for representative intents
# - The "unchanged_intents" list should be minimal - only for intents that are completely unique
# - Aim to reduce the total number of distinct intents significantly through merging"""

# def get_system_prompt():
#     """定义系统提示词"""
#     return """You are an expert in intent analysis and text processing. Your task is to aggressively merge similar intents within a given group into broader, coarse-grained categories.
#
# Rules:
# 1. PRIORITIZE MERGING - Always look for opportunities to combine intents into broader categories
# 2. Merge intents that share similar themes, domains, or purposes, even if they have different specific expressions
# 3. Create coarse-grained, high-level intent categories that encompass multiple related specific intents
# 4. For merged intents, choose a CONCISE SINGLE WORD OR SHORT PHRASE (maximum 3-4 words) that covers all merged intents
# 5. Only keep intents separate if they belong to completely different domains or have fundamentally different purposes
# 6. When in doubt, merge rather than keep separate
# 7. Output must be in strict JSON format
#
# Merging Guidelines:
# - Merge intents with similar semantic meaning or purpose
# - Merge intents from the same domain or category (e.g., all shopping-related, all information-seeking, etc.)
# - Merge intents that represent different ways of expressing the same underlying need
# - Create broader umbrella categories that can encompass multiple specific use cases
# - Aim for fewer, more comprehensive intent categories rather than many specific ones
#
# Intent Naming Requirements:
# - Representative intents MUST be concise: single words or short phrases only
# - Maximum 3-4 words per intent name
# - Use simple, clear terminology (e.g., "Shopping", "Information Request", "Technical Support")
# - Avoid full sentences, explanations, or verbose descriptions
# - Use generic category names that broadly capture the intent group
#
# Output format:
# {
#     "merged_intents": [
#         {
#             "representative_intent": "concise category name (1-4 words max)",
#             "merged_from": ["intent1", "intent2", "intent3", "intent4"]
#         }
#     ],
#     "unchanged_intents": ["only truly unique intents that cannot be merged"]
# }
#
# Important:
# - Favor aggressive merging - err on the side of combining rather than separating
# - Each intent must appear in exactly one place (either as merged or unchanged)
# - Representative intent names must be SHORT and GENERIC category labels
# - The "unchanged_intents" list should be minimal - only for intents that are completely unique
# - Aim to reduce the total number of distinct intents significantly through merging
# - Examples of good representative intents: "Literature", "Philosophy", "Psychology", "Comedy"
# - Strictly adhere to the output format above, please do not give reasons or Rationale for the merge
# """

def get_system_prompt():
    """Define system prompt for intent merging."""
    return """You are an expert in intent analysis and text processing. Your task is to aggressively merge similar intents into broader categories.

Instructions:
1. Merge intents that share similar themes, domains, or purposes.
2. Create high-level categories (maximum 3-4 words) that encompass related intents.
3. Only keep intents separate if they belong to completely different domains.
4. Always prioritize merging; merge rather than keep separate when in doubt.
5. Do not give any Explanation
6. Output must be in strict JSON format.

Output format:
{
    "merged_intents": [
        {
            "representative_intent": "short category name",
            "merged_from": ["intent1", "intent2", "intent3"]
        }
    ],
    "unchanged_intents": ["unique intent1", "unique intent2"]
}
"""

def get_user_prompt(intents_in_cluster):
    """生成用户提示词"""
    intents_str = '\n'.join([f"- {intent}" for intent in intents_in_cluster])

    prompt = f"""Please analyze the following intents and merge those that are semantically similar:

{intents_str}

Please identify which intents express essentially the same meaning and can be merged together. Be conservative - only merge intents that are truly similar in meaning."""

    return prompt


def prepare_input_for_cluster(cluster_id, intents_in_cluster):
    """为一个聚类准备输入"""
    message = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": get_user_prompt(intents_in_cluster)}
    ]

    row = {
        "custom_id": f"cluster_{cluster_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": llm,
            "messages": message,
            "max_tokens": 2000,
            "temperature": 0.1  # 降低随机性，保证输出稳定
        }
    }

    return row


def create_batch_input(output_dict, min_cluster_size=2):
    """
    创建批处理输入文件

    Args:
        output_dict: 意图到聚类ID的映射字典 {intent: cluster_id}
        min_cluster_size: 最小聚类大小，小于此大小的聚类不进行合并
    """

    # 按聚类ID分组意图
    cluster_to_intents = defaultdict(list)
    for intent, cluster_id in output_dict.items():
        cluster_to_intents[cluster_id].append(intent)

    # 统计信息
    print(f"总共有 {len(cluster_to_intents)} 个聚类")
    print(f"聚类大小分布:")
    size_dist = defaultdict(int)
    for cluster_id, intents in cluster_to_intents.items():
        size_dist[len(intents)] += 1
    for size in sorted(size_dist.keys()):
        print(f"  大小为 {size} 的聚类: {size_dist[size]} 个")

    # 准备输入数据
    input_content = []
    processed_clusters = 0
    skipped_clusters = 0

    for cluster_id, intents_in_cluster in tqdm(cluster_to_intents.items(), desc='准备输入数据'):
        if len(intents_in_cluster) < min_cluster_size:
            print(f"跳过聚类 {cluster_id}，大小为 {len(intents_in_cluster)} (小于最小大小 {min_cluster_size})")
            skipped_clusters += 1
            continue

        # 去重并排序
        intents_in_cluster = sorted(list(set(intents_in_cluster)))

        print(f"处理聚类 {cluster_id}，包含 {len(intents_in_cluster)} 个意图:")
        for intent in intents_in_cluster[:5]:  # 显示前5个
            print(f"  - {intent}")
        if len(intents_in_cluster) > 5:
            print(f"  ... 还有 {len(intents_in_cluster) - 5} 个意图")

        row = prepare_input_for_cluster(cluster_id, intents_in_cluster)
        input_content.append(row)
        processed_clusters += 1

    print(f"\n处理统计:")
    print(f"  处理的聚类数: {processed_clusters}")
    print(f"  跳过的聚类数: {skipped_clusters}")
    print(f"  将调用LLM {len(input_content)} 次")

    # 确保输出目录存在
    os.makedirs('batch_input', exist_ok=True)

    # 写入jsonl文件
    output_file = f'batch_input/{dataset}/{dataset}_first_intent_merge_{llm}_input.jsonl'
    with jsonlines.open(output_file, mode='w') as writer:
        for row in input_content:
            writer.write(row)

    print(f"\n批处理输入文件已保存到: {output_file}")
    return output_file, processed_clusters


def validate_input_file(file_path):
    """验证生成的输入文件"""
    print(f"\n验证输入文件: {file_path}")

    try:
        with jsonlines.open(file_path, mode='r') as reader:
            count = 0
            for row in reader:
                count += 1
                # 验证必要字段
                assert 'custom_id' in row
                assert 'method' in row
                assert 'url' in row
                assert 'body' in row
                assert 'model' in row['body']
                assert 'messages' in row['body']

            print(f"文件验证成功！共 {count} 条记录")

            # 显示第一条记录作为示例
            with jsonlines.open(file_path, mode='r') as reader:
                first_row = next(reader)
                print(f"\n示例记录:")
                print(f"Custom ID: {first_row['custom_id']}")
                print(f"Model: {first_row['body']['model']}")
                print(f"System prompt 长度: {len(first_row['body']['messages'][0]['content'])}")
                print(f"User prompt 长度: {len(first_row['body']['messages'][1]['content'])}")

    except Exception as e:
        print(f"文件验证失败: {e}")
        return False

    return True

output_file, processed_clusters = create_batch_input(output_dict, min_cluster_size=2)