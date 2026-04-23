import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import re
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

# 下载必要的NLTK资源
nltk.download('stopwords', quiet=True)

# 文本预处理函数
def preprocess_text(text):
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 处理否定词，确保它们被保留
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'nt", " not", text)
    # 移除标点符号，但保留必要的空格
    text = re.sub(r"[^a-zA-Z\s]", ' ', text)
    # 分词（简单分词，保留短语）
    words = text.split()
    # 移除停用词（保留否定词）
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'isn', 'wasn', 'weren', 'don', 'didn', 'doesn', 'haven', 'hasn', 'hadn', 'won', 'wouldn', 'shouldn', 'couldn', 'mightn'}
    stop_words = stop_words - negation_words
    # 过滤停用词
    words = [word for word in words if word not in stop_words]
    return ' '.join(words), words

# 读取训练数据
print("Reading training data...")
train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')

# 读取测试数据
print("Reading test data...")
test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

# 预处理训练数据
print("Preprocessing training data...")
train_data['clean_review'], train_data['clean_words'] = zip(*train_data['review'].apply(preprocess_text))

# 预处理测试数据
print("Preprocessing test data...")
test_data['clean_review'], test_data['clean_words'] = zip(*test_data['review'].apply(preprocess_text))

# 提取TF-IDF特征（优化的短语模式）
print("Extracting TF-IDF features with optimized phrase patterns...")
vectorizer = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1, 3),  # 短语模式：1-3gram
    min_df=2,
    max_df=0.8,
    sublinear_tf=True,
    use_idf=True,
    smooth_idf=True
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
    negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'isn', 'wasn', 'weren', 'don', 'didn', 'doesn', 'haven', 'hasn', 'hadn', 'won', 'wouldn', 'shouldn', 'couldn', 'mightn'}
    return sum(1 for word in words if word in negation_words)

train_data['negation_count'] = train_data['clean_words'].apply(count_negation_words)
test_data['negation_count'] = test_data['clean_words'].apply(count_negation_words)

# 平均词长特征
def avg_word_length(words):
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

train_data['avg_word_length'] = train_data['clean_words'].apply(avg_word_length)
test_data['avg_word_length'] = test_data['clean_words'].apply(avg_word_length)

# 标准化额外特征
scaler = StandardScaler()
extra_features_train = scaler.fit_transform(train_data[['review_length', 'word_count', 'negation_count', 'avg_word_length']])
extra_features_test = scaler.transform(test_data[['review_length', 'word_count', 'negation_count', 'avg_word_length']])

# 转换为稀疏矩阵
extra_features_train_sparse = csr_matrix(extra_features_train)
extra_features_test_sparse = csr_matrix(extra_features_test)

# 合并特征
print("Merging features...")
X_train_combined = hstack([X_train_tfidf, extra_features_train_sparse])
X_test_combined = hstack([X_test_tfidf, extra_features_test_sparse])

# 调优逻辑回归模型
print("Tuning Logistic Regression model...")
param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'penalty': ['l2'],
    'solver': ['saga', 'liblinear']
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=3000, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_combined, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation AUC: {grid_search.best_score_:.6f}")

# 定义其他模型
print("Defining other models...")

# 岭回归分类器
ridge_model = RidgeClassifier(
    alpha=1.0,
    random_state=42
)

# 逻辑回归模型（使用最佳参数）
best_lr_model = grid_search.best_estimator_

# 创建集成模型
print("Creating ensemble model...")
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', best_lr_model),
        ('ridge', ridge_model)
    ],
    voting='soft',
    n_jobs=-1
)

# 训练集成模型
print("Training ensemble model...")
ensemble_model.fit(X_train_combined, y_train)

# 验证模型性能
print("Evaluating model performance...")
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42
)

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
