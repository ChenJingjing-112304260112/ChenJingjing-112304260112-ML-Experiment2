import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# 高级文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 保留否定词的特殊处理
    text = re.sub(r"n't", " not", text)
    # 去除标点和特殊符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 预处理训练数据
print("Preprocessing training data...")
train_texts = [preprocess_text(text) for text in train_df['review']]

# 预处理测试数据
print("Preprocessing test data...")
test_texts = [preprocess_text(text) for text in test_df['review']]

# 生成TF-IDF特征（优化参数）
print("Generating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=20000,  # 增加特征数量
    ngram_range=(1, 3),  # 包含1-gram、2-gram和3-gram
    stop_words='english',
    min_df=2,
    max_df=0.8
)

train_features = tfidf.fit_transform(train_texts)
test_features = tfidf.transform(test_texts)

# 添加额外特征
print("Adding additional features...")

# 文本长度特征
train_length = np.array([len(text) for text in train_texts]).reshape(-1, 1)
test_length = np.array([len(text) for text in test_texts]).reshape(-1, 1)

# 词数特征
train_word_count = np.array([len(text.split()) for text in train_texts]).reshape(-1, 1)
test_word_count = np.array([len(text.split()) for text in test_texts]).reshape(-1, 1)

# 否定词数量特征
def count_negations(text):
    neg_words = ['not', 'no', 'never', 'nothing', 'nowhere', 'noone', 'none', 'cannot']
    return sum(1 for word in text.split() if word in neg_words)

train_neg = np.array([count_negations(text) for text in train_texts]).reshape(-1, 1)
test_neg = np.array([count_negations(text) for text in test_texts]).reshape(-1, 1)

# 合并所有特征
train_features = hstack([train_features, train_length, train_word_count, train_neg])
test_features = hstack([test_features, test_length, test_word_count, test_neg])

# 准备训练数据
y_train = train_df['sentiment'].values

# 划分验证集
X_train, X_val, y_train_sub, y_val = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# 训练多个模型
print("Training ensemble models...")

# 模型1：逻辑回归
lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

# 模型2：随机森林
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 模型3：梯度提升
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

# 集成模型
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft',  # 使用概率投票
    n_jobs=-1
)

# 训练集成模型
ensemble_model.fit(X_train, y_train_sub)

# 在验证集上评估
val_pred = ensemble_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")

# 在测试集上预测
print("Predicting on test data...")
test_pred = ensemble_model.predict_proba(test_features)[:, 1]

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': (test_pred > 0.5).astype(int)
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")