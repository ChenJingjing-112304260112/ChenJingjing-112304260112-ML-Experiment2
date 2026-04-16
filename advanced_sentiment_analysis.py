import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 更高级的文本预处理函数（保留否定词和短语）
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

# 训练Word2Vec模型（优化参数）
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    sentences=train_tokens,
    vector_size=200,  # 增加向量维度
    window=7,  # 增加窗口大小
    min_count=3,  # 减少最小词频
    workers=4,
    epochs=20,  # 增加训练轮数
    sg=1  # 使用skip-gram模型
)

# 句子向量表示（改进版：加权平均）
def get_sentence_vector(tokens, model):
    vectors = []
    weights = []
    for i, token in enumerate(tokens):
        if token in model.wv:
            # 位置权重：句子开头和结尾的词更重要
            weight = 1.0
            if i == 0 or i == len(tokens) - 1:
                weight = 1.5
            vectors.append(model.wv[token] * weight)
            weights.append(weight)
    if not vectors:
        return np.zeros(model.vector_size)
    return np.sum(vectors, axis=0) / np.sum(weights)

# 生成训练集向量
print("Generating training vectors...")
train_vectors = []
for tokens in train_tokens:
    train_vectors.append(get_sentence_vector(tokens, word2vec_model))
train_vectors = np.array(train_vectors)

# 生成测试集向量
print("Generating test vectors...")
test_vectors = []
for tokens in test_tokens:
    test_vectors.append(get_sentence_vector(tokens, word2vec_model))
test_vectors = np.array(test_vectors)

# 准备训练数据
y_train = train_df['sentiment'].values

# 划分验证集
X_train, X_val, y_train_sub, y_val = train_test_split(train_vectors, y_train, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_vectors_scaled = scaler.transform(test_vectors)

# 训练随机森林模型（优化参数）
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,  # 增加树的数量
    max_depth=50,  # 增加树的深度
    min_samples_split=5,  # 减少最小分裂样本数
    min_samples_leaf=2,  # 减少最小叶节点样本数
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
test_pred = model.predict_proba(test_vectors_scaled)[:, 1]

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': (test_pred > 0.5).astype(int)
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")