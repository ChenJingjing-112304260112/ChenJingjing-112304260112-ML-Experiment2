import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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

# 训练Word2Vec模型（内存高效版本）
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    sentences=train_tokens,
    vector_size=150,
    window=5,
    min_count=5,
    workers=4,
    epochs=15,
    sg=0  # 使用CBOW模型，内存更高效
)

# 句子向量表示
def get_sentence_vector(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 生成Word2Vec特征
print("Generating Word2Vec features...")
train_w2v = []
for tokens in train_tokens:
    train_w2v.append(get_sentence_vector(tokens, word2vec_model))
train_w2v = np.array(train_w2v)

test_w2v = []
for tokens in test_tokens:
    test_w2v.append(get_sentence_vector(tokens, word2vec_model))
test_w2v = np.array(test_w2v)

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
train_features = np.hstack([train_w2v, train_length, train_neg])
test_features = np.hstack([test_w2v, test_length, test_neg])

# 准备训练数据
y_train = train_df['sentiment'].values

# 划分验证集
X_train, X_val, y_train_sub, y_val = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_features_scaled = scaler.transform(test_features)

# 训练逻辑回归模型（优化参数）
print("Training Logistic Regression model...")
model = LogisticRegression(
    C=0.1,  # 正则化参数
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train_sub)

# 在验证集上评估
val_pred = model.predict_proba(X_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")

# 在测试集上预测
print("Predicting on test data...")
test_pred = model.predict_proba(test_features_scaled)[:, 1]

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': (test_pred > 0.5).astype(int)
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")