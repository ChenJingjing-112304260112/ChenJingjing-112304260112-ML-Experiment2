import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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
    # 去除标点和特殊符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 读取训练数据
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)

# 读取测试数据
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 预处理训练数据
train_texts = [preprocess_text(text) for text in train_df['review']]

# 预处理测试数据
test_texts = [preprocess_text(text) for text in test_df['review']]

# 生成TF-IDF特征
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words='english'
)

train_features = tfidf.fit_transform(train_texts)
test_features = tfidf.transform(test_texts)

# 添加额外特征
# 文本长度特征
train_length = np.array([len(text) for text in train_texts]).reshape(-1, 1)
test_length = np.array([len(text) for text in test_texts]).reshape(-1, 1)

# 词数特征
train_word_count = np.array([len(text.split()) for text in train_texts]).reshape(-1, 1)
test_word_count = np.array([len(text.split()) for text in test_texts]).reshape(-1, 1)

# 合并所有特征
train_features = hstack([train_features, train_length, train_word_count])
test_features = hstack([test_features, test_length, test_word_count])

# 准备训练数据
y_train = train_df['sentiment'].values

# 划分验证集
X_train, X_val, y_train_sub, y_val = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = XGBClassifier(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective='binary:logistic',
    eval_metric='auc'
)

model.fit(
    X_train, y_train_sub,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=True
)

# 在验证集上评估
val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")

# 在测试集上预测
test_pred = model.predict_proba(test_features)[:, 1]

# 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': (test_pred > 0.5).astype(int)
})

# 保存提交文件
submission_df.to_csv('submission.csv', index=False)
print("Submission file generated with validation AUC: {:.4f}".format(val_auc))