import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english'
)

train_features = tfidf.fit_transform(train_texts)
test_features = tfidf.transform(test_texts)

# 准备训练数据
y_train = train_df['sentiment'].values

# 划分验证集
X_train, X_val, y_train_sub, y_val = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# 训练多个模型
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
    random_state=42,
    n_jobs=-1
)

# 模型3：XGBoost
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# 集成模型
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    voting='soft',
    n_jobs=-1
)

# 训练集成模型
ensemble_model.fit(X_train, y_train_sub)

# 在验证集上评估
val_pred = ensemble_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")

# 在测试集上预测
test_pred = ensemble_model.predict_proba(test_features)[:, 1]

# 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': (test_pred > 0.5).astype(int)
})

# 保存提交文件
submission_df.to_csv('submission.csv', index=False)
print("Submission file generated with validation AUC: {:.4f}".format(val_auc))