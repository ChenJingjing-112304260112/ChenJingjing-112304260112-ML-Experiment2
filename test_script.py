import pandas as pd
import numpy as np
import re

# 简单的文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 去除标点和特殊符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = text.split()
    # 简单的停用词过滤
    stop_words = set(['a', 'an', 'the', 'and', 'or', 'but'])
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# 读取少量训练数据进行测试
print("Reading test data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3, nrows=100)

# 预处理文本
print("Preprocessing text...")
tokens = [preprocess_text(text) for text in train_df['review']]

print(f"Processed {len(tokens)} reviews")
print(f"First review tokens: {tokens[0][:10]}...")

print("Test completed successfully!")