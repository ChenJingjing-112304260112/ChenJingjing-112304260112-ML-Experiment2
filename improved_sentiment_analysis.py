import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 文本预处理函数（保留否定词）
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 去除标点和特殊符号（保留否定词相关的符号）
    text = re.sub(r'[^a-zA-Z\s\'t]', '', text)
    # 分词
    tokens = text.split()
    # 自定义停用词列表（不包含否定词）
    stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on'])
    # 保留否定词
    tokens = [word for word in tokens if word not in stop_words or word.endswith("n't") or word == "not"]
    return tokens

# 读取数据
print("Reading data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
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

# 训练Word2Vec模型
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    sentences=train_tokens,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=10
)

# 保存Word2Vec模型
word2vec_model.save('word2vec_model.model')
print("Word2Vec model saved: word2vec_model.model")

# 句子向量表示（平均词向量）
def get_sentence_vector(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 生成训练集向量
print("Generating training vectors...")
train_vectors = []
for tokens in train_tokens:
    train_vectors.append(get_sentence_vector(tokens, word2vec_model))
train_vectors = np.array(train_vectors)

# 保存训练集向量
np.save('train_vectors.npy', train_vectors)
print("Training vectors saved: train_vectors.npy")

# 生成测试集向量
print("Generating test vectors...")
test_vectors = []
for tokens in test_tokens:
    test_vectors.append(get_sentence_vector(tokens, word2vec_model))
test_vectors = np.array(test_vectors)

# 保存测试集向量
np.save('test_vectors.npy', test_vectors)
print("Test vectors saved: test_vectors.npy")

# 准备训练数据
y_train = train_df['sentiment'].values

# 划分验证集
X_train, X_val, y_train_sub, y_val = train_test_split(train_vectors, y_train, test_size=0.2, random_state=42)

# 训练随机森林模型
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train_sub)

# 在验证集上评估
val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")

# 在测试集上预测
print("Predicting on test data...")
test_pred = model.predict_proba(test_vectors)[:, 1]

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': (test_pred > 0.5).astype(int)
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")