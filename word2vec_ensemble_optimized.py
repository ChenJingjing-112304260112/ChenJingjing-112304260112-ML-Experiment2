import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 下载必要的NLTK资源
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# 文本预处理函数
def preprocess_text(text):
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 移除标点符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    words = text.split()
    # 移除停用词（保留否定词）
    stop_words = set(stopwords.words('english'))
    # 保留否定词
    negation_words = {'not', "n't", 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor'}
    stop_words = stop_words - negation_words
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# 读取训练数据
train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')

# 预处理训练数据
train_data['clean_words'] = train_data['review'].apply(preprocess_text)

# 读取测试数据
test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

# 预处理测试数据
test_data['clean_words'] = test_data['review'].apply(preprocess_text)

# 训练Word2Vec模型
print("Training Word2Vec model...")
all_words = train_data['clean_words'].tolist() + test_data['clean_words'].tolist()
word2vec_model = Word2Vec(
    sentences=all_words,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)

# 生成Word2Vec特征
def get_word2vec_features(words, model, vector_size):
    feature_vector = np.zeros(vector_size)
    count = 0
    for word in words:
        if word in model.wv:
            feature_vector += model.wv[word]
            count += 1
    if count > 0:
        feature_vector /= count
    return feature_vector

print("Generating Word2Vec features...")
X_train_w2v = np.array([get_word2vec_features(words, word2vec_model, 300) for words in train_data['clean_words']])
X_test_w2v = np.array([get_word2vec_features(words, word2vec_model, 300) for words in test_data['clean_words']])

# 生成TF-IDF特征
print("Generating TF-IDF features...")
train_data['clean_review'] = train_data['clean_words'].apply(' '.join)
test_data['clean_review'] = test_data['clean_words'].apply(' '.join)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(train_data['clean_review'])
X_test_tfidf = vectorizer.transform(test_data['clean_review'])

# 合并特征
print("Merging features...")
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

# 标准化Word2Vec特征
scaler = StandardScaler()
X_train_w2v_scaled = scaler.fit_transform(X_train_w2v)
X_test_w2v_scaled = scaler.transform(X_test_w2v)

# 转换为稀疏矩阵
from scipy.sparse import csr_matrix
X_train_w2v_sparse = csr_matrix(X_train_w2v_scaled)
X_test_w2v_sparse = csr_matrix(X_test_w2v_scaled)

# 合并特征
X_train_combined = hstack([X_train_tfidf, X_train_w2v_sparse])
X_test_combined = hstack([X_test_tfidf, X_test_w2v_sparse])
y_train = train_data['sentiment']

# 定义分类模型
print("Training models...")

# 随机森林模型（Word2Vec特征）
rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# XGBoost模型（组合特征）
xgb_model = XGBClassifier(
    n_estimators=1500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 逻辑回归模型（TF-IDF特征）
lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    random_state=42,
    max_iter=1000
)

# 创建集成模型
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft',
    n_jobs=-1
)

# 训练集成模型
ensemble_model.fit(X_train_combined, y_train)

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

# 验证模型性能
print("Evaluating model performance...")
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42
)

# 训练验证模型
val_model = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )),
        ('rf', RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ('lr', LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        ))
    ],
    voting='soft',
    n_jobs=-1
)

val_model.fit(X_train_split, y_train_split)
y_val_pred = val_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC score: {val_auc}")
