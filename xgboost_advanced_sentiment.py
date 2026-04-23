import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

# 高级文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 保留否定词的特殊处理
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"aren't", "are not", text)
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
    ngram_range=(1, 2),  # 包含1-gram和2-gram
    stop_words='english',
    min_df=2,  # 最小文档频率
    max_df=0.8,  # 最大文档频率
    lowercase=False  # 已经在预处理中转为小写
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
    neg_words = ['not', 'no', 'never', 'nothing', 'nowhere', 'noone', 'none', 'cannot', 'nevertheless', 'without', 'hardly', 'scarcely', 'barely', 'seldom', 'rarely']
    return sum(1 for word in text.split() if word in neg_words)

train_neg = np.array([count_negations(text) for text in train_texts]).reshape(-1, 1)
test_neg = np.array([count_negations(text) for text in test_texts]).reshape(-1, 1)

# 情感词数量特征
def count_sentiment_words(text):
    positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'awesome', 'love', 'like', 'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'terrific', 'enjoyable', 'entertaining', 'impressive', 'beautiful', 'excellent', 'wonderful', 'fantastic', 'amazing', 'incredible', 'marvelous', 'splendid', 'magnificent', 'exceptional', 'remarkable']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'poor', 'boring', 'dreadful', 'pathetic', 'lousy', 'horrendous', 'disgusting', 'hate', 'dislike', 'worse', 'awful', 'terrible', 'negative', 'horrible', 'dreadful', 'appalling', 'abysmal', 'atrocious', 'awful', 'disastrous', 'dismal', 'horrendous', 'miserable', 'pathetic']
    pos_count = sum(1 for word in text.split() if word in positive_words)
    neg_count = sum(1 for word in text.split() if word in negative_words)
    return pos_count, neg_count

train_pos = np.array([count_sentiment_words(text)[0] for text in train_texts]).reshape(-1, 1)
test_pos = np.array([count_sentiment_words(text)[0] for text in test_texts]).reshape(-1, 1)
train_neg_sentiment = np.array([count_sentiment_words(text)[1] for text in train_texts]).reshape(-1, 1)
test_neg_sentiment = np.array([count_sentiment_words(text)[1] for text in test_texts]).reshape(-1, 1)

# 词多样性特征
def word_diversity(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

train_diversity = np.array([word_diversity(text) for text in train_texts]).reshape(-1, 1)
test_diversity = np.array([word_diversity(text) for text in test_texts]).reshape(-1, 1)

# 合并所有特征
train_features = hstack([train_features, train_length, train_word_count, train_neg, train_pos, train_neg_sentiment, train_diversity])
test_features = hstack([test_features, test_length, test_word_count, test_neg, test_pos, test_neg_sentiment, test_diversity])

# 准备训练数据
y_train = train_df['sentiment'].values

# 划分验证集
X_train, X_val, y_train_sub, y_val = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# 训练XGBoost模型（优化参数）
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=1000,  # 增加树的数量
    max_depth=7,  # 树的深度
    learning_rate=0.05,  # 学习率
    subsample=0.8,  # 子采样比例
    colsample_bytree=0.8,  # 列采样比例
    gamma=0.1,  # 分裂阈值
    reg_alpha=0.1,  # L1正则化
    reg_lambda=1.0,  # L2正则化
    random_state=42,
    n_jobs=-1,
    objective='binary:logistic',
    eval_metric='auc'
)

# 训练模型
model.fit(
    X_train, y_train_sub,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    early_stopping_rounds=100,  # 早停
    verbose=True
)

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