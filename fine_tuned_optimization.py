import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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
    # 词干提取
    stemmer = PorterStemmer()
    # 过滤停用词并进行词干提取
    words = [stemmer.stem(word) for word in words if word not in stop_words]
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

# 提取TF-IDF特征（精细调优）
print("Extracting TF-IDF features with fine-tuned parameters...")
vectorizer = TfidfVectorizer(
    max_features=70000,
    ngram_range=(1, 3),  # 短语模式：1-3gram
    min_df=3,
    max_df=0.75,
    sublinear_tf=True,
    use_idf=True,
    smooth_idf=True,
    strip_accents='unicode'
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
    # 检查词干形式
    stemmer = PorterStemmer()
    negation_stems = {stemmer.stem(word) for word in negation_words}
    return sum(1 for word in words if word in negation_stems)

train_data['negation_count'] = train_data['clean_words'].apply(count_negation_words)
test_data['negation_count'] = test_data['clean_words'].apply(count_negation_words)

# 平均词长特征
def avg_word_length(words):
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

train_data['avg_word_length'] = train_data['clean_words'].apply(avg_word_length)
test_data['avg_word_length'] = test_data['clean_words'].apply(avg_word_length)

# 单词多样性特征
def word_diversity(words):
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

train_data['word_diversity'] = train_data['clean_words'].apply(word_diversity)
test_data['word_diversity'] = test_data['clean_words'].apply(word_diversity)

# 标准化额外特征
scaler = StandardScaler()
extra_features_train = scaler.fit_transform(train_data[['review_length', 'word_count', 'negation_count', 'avg_word_length', 'word_diversity']])
extra_features_test = scaler.transform(test_data[['review_length', 'word_count', 'negation_count', 'avg_word_length', 'word_diversity']])

# 转换为稀疏矩阵
extra_features_train_sparse = csr_matrix(extra_features_train)
extra_features_test_sparse = csr_matrix(extra_features_test)

# 合并特征
print("Merging features...")
X_train_combined = hstack([X_train_tfidf, extra_features_train_sparse])
X_test_combined = hstack([X_test_tfidf, extra_features_test_sparse])

# 调优逻辑回归模型（更精细的参数网格）
print("Tuning Logistic Regression model with fine-grained parameters...")
param_grid = {
    'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    'penalty': ['l2'],
    'solver': ['saga'],
    'max_iter': [3000]
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_combined, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation AUC: {grid_search.best_score_:.6f}")

# 使用最佳参数训练模型
print("Training model with best parameters...")
best_model = grid_search.best_estimator_

# 验证模型性能
print("Evaluating model performance...")
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42
)

y_val_pred = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC score: {val_auc:.6f}")

# 预测测试数据
print("Predicting test data...")
y_pred = best_model.predict(X_test_combined)

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
