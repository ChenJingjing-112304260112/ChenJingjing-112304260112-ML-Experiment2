import pandas as pd
import numpy as np
import re

# 文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 保留否定词的特殊处理
    text = re.sub(r"n't", " not", text)
    # 去除标点和特殊符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = text.split()
    # 自定义停用词列表（不包含否定词）
    stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on'])
    # 保留否定词
    tokens = [word for word in tokens if word not in stop_words or word == "not"]
    return tokens

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 预处理训练数据
print("Preprocessing training data...")
train_tokens = []
for text in train_df['review']:
    train_tokens.append(preprocess_text(text))

# 预处理测试数据
print("Preprocessing test data...")
test_tokens = []
for text in test_df['review']:
    test_tokens.append(preprocess_text(text))

# 简单的词袋模型特征
print("Generating bag-of-words features...")

# 构建词汇表
vocab = {}
for tokens in train_tokens:
    for token in tokens:
        if token in vocab:
            vocab[token] += 1
        else:
            vocab[token] = 1

# 过滤低频词
filtered_vocab = {word: count for word, count in vocab.items() if count >= 5}
word_to_idx = {word: i for i, word in enumerate(filtered_vocab.keys())}

# 生成词袋特征
def get_bag_of_words(tokens, word_to_idx):
    vec = np.zeros(len(word_to_idx))
    for token in tokens:
        if token in word_to_idx:
            vec[word_to_idx[token]] += 1
    return vec

train_bow = []
for tokens in train_tokens:
    train_bow.append(get_bag_of_words(tokens, word_to_idx))
train_bow = np.array(train_bow)

test_bow = []
for tokens in test_tokens:
    test_bow.append(get_bag_of_words(tokens, word_to_idx))
test_bow = np.array(test_bow)

# 添加额外特征
print("Adding additional features...")

# 文本长度特征
train_length = np.array([len(tokens) for tokens in train_tokens]).reshape(-1, 1)
test_length = np.array([len(tokens) for tokens in test_tokens]).reshape(-1, 1)

# 否定词数量特征
def count_negations(tokens):
    neg_words = ['not', 'no', 'never', 'nothing', 'nowhere', 'noone', 'none', 'cannot']
    return sum(1 for token in tokens if token in neg_words)

train_neg = np.array([count_negations(tokens) for tokens in train_tokens]).reshape(-1, 1)
test_neg = np.array([count_negations(tokens) for tokens in test_tokens]).reshape(-1, 1)

# 合并所有特征
train_features = np.hstack([train_bow, train_length, train_neg])
test_features = np.hstack([test_bow, test_length, test_neg])

# 准备训练数据
y_train = train_df['sentiment'].values

# 划分验证集
from sklearn.model_selection import train_test_split
X_train, X_val, y_train_sub, y_val = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# 训练逻辑回归模型
print("Training Logistic Regression model...")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_sub)

# 在验证集上评估
val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")

# 在测试集上预测
print("Predicting on test data...")
test_pred = model.predict_proba(test_features)[:, 1]

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': (test_pred > 0.5).astype(int)
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")