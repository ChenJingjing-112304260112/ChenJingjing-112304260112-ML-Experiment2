import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

# 下载必要的NLTK资源
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 文本预处理函数
def preprocess_text(text):
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 提取英文单词，保留带缩写的否定形式
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    # 分词
    words = text.split()
    # 移除停用词（保留否定词）
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', "n't", 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor'}
    stop_words = stop_words - negation_words
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# 读取训练数据
print("Reading training data...")
train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')

# 读取测试数据
print("Reading test data...")
test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

# 预处理训练数据
print("Preprocessing training data...")
train_data['clean_words'] = train_data['review'].apply(preprocess_text)
train_data['clean_review'] = train_data['clean_words'].apply(' '.join)

# 预处理测试数据
print("Preprocessing test data...")
test_data['clean_words'] = test_data['review'].apply(preprocess_text)
test_data['clean_review'] = test_data['clean_words'].apply(' '.join)

# 提取TF-IDF特征（优化参数）
print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=40000,  # 增加特征数量
    ngram_range=(1, 3),  # 增加ngram范围
    min_df=2,
    max_df=0.9,
    sublinear_tf=True  # 亚线性TF缩放
)

X_train_tfidf = vectorizer.fit_transform(train_data['clean_review'])
y_train = train_data['sentiment']
X_test_tfidf = vectorizer.transform(test_data['clean_review'])

# 提取额外特征
print("Extracting additional features...")

# 文本长度特征
train_data['review_length'] = train_data['clean_review'].apply(len)
test_data['review_length'] = test_data['clean_review'].apply(len)

# 单词数量特征
train_data['word_count'] = train_data['clean_words'].apply(len)
test_data['word_count'] = test_data['clean_words'].apply(len)

# 否定词数量特征
def count_negation_words(words):
    negation_words = {'not', "n't", 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor'}
    return sum(1 for word in words if word in negation_words)

train_data['negation_count'] = train_data['clean_words'].apply(count_negation_words)
test_data['negation_count'] = test_data['clean_words'].apply(count_negation_words)

# 标准化额外特征
scaler = StandardScaler()
extra_features_train = scaler.fit_transform(train_data[['review_length', 'word_count', 'negation_count']])
extra_features_test = scaler.transform(test_data[['review_length', 'word_count', 'negation_count']])

# 转换为稀疏矩阵
extra_features_train_sparse = csr_matrix(extra_features_train)
extra_features_test_sparse = csr_matrix(extra_features_test)

# 合并特征
print("Merging features...")
X_train_combined = hstack([X_train_tfidf, extra_features_train_sparse])
X_test_combined = hstack([X_test_tfidf, extra_features_test_sparse])

# 拆分训练集和验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42
)

# 定义基础模型
print("Training base models...")

# 逻辑回归模型
lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='saga',
    random_state=42,
    max_iter=1000,
    n_jobs=-1
)

# XGBoost模型
xgb_model = XGBClassifier(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 创建集成模型
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    voting='soft',
    n_jobs=-1
)

# 训练集成模型
print("Training ensemble model...")
ensemble_model.fit(X_train_combined, y_train)

# 验证模型性能
print("Evaluating model performance...")
y_val_pred = ensemble_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC score: {val_auc:.6f}")

# 预测测试数据
print("Predicting test data...")
y_pred = ensemble_model.predict(X_test_combined)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_data['id'],
    'sentiment': y_pred
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(f"Submission shape: {submission.shape}")
print(f"First few rows:\n{submission.head()}")
