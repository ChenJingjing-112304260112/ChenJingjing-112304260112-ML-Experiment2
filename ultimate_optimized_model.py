import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

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
    # 提取英文单词，保留带缩写的否定形式
    text = re.sub(r"[^a-zA-Z\s']", '', text)
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
print("Reading training data...")
train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')

# 读取未标记数据
print("Reading unlabeled training data...")
unlabeled_train_data = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t')

# 读取测试数据
print("Reading test data...")
test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

# 预处理训练数据
print("Preprocessing training data...")
train_data['clean_words'] = train_data['review'].apply(preprocess_text)
train_data['clean_review'] = train_data['clean_words'].apply(' '.join)

# 预处理未标记数据
print("Preprocessing unlabeled training data...")
unlabeled_train_data['clean_words'] = unlabeled_train_data['review'].apply(preprocess_text)
unlabeled_train_data['clean_review'] = unlabeled_train_data['clean_words'].apply(' '.join)

# 预处理测试数据
print("Preprocessing test data...")
test_data['clean_words'] = test_data['review'].apply(preprocess_text)
test_data['clean_review'] = test_data['clean_words'].apply(' '.join)

# 训练Word2Vec模型
print("Training Word2Vec model...")
all_words = train_data['clean_words'].tolist() + \
            unlabeled_train_data['clean_words'].tolist() + \
            test_data['clean_words'].tolist()

word2vec_model = Word2Vec(
    sentences=all_words,
    vector_size=300,
    window=5,
    min_count=5,
    workers=4,
    sg=1,  # Skip-gram
    hs=0,  # Negative Sampling
    negative=5,
    sample=0.001,
    epochs=20
)

# 生成Word2Vec特征（平均词向量）
def get_mean_embedding(words, model, vector_size):
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
X_train_w2v = np.array([get_mean_embedding(words, word2vec_model, 300) for words in train_data['clean_words']])
X_test_w2v = np.array([get_mean_embedding(words, word2vec_model, 300) for words in test_data['clean_words']])

# 生成TF-IDF特征
print("Generating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(train_data['clean_review'])
X_test_tfidf = vectorizer.transform(test_data['clean_review'])

# 合并特征
print("Merging features...")
# 标准化Word2Vec特征
scaler = StandardScaler()
X_train_w2v_scaled = scaler.fit_transform(X_train_w2v)
X_test_w2v_scaled = scaler.transform(X_test_w2v)

# 转换为稀疏矩阵
X_train_w2v_sparse = csr_matrix(X_train_w2v_scaled)
X_test_w2v_sparse = csr_matrix(X_test_w2v_scaled)

# 合并特征
X_train_combined = hstack([X_train_tfidf, X_train_w2v_sparse])
X_test_combined = hstack([X_test_tfidf, X_test_w2v_sparse])
y_train = train_data['sentiment']

# 分层5折交叉验证评估
print("Performing 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义模型
lr_model = LogisticRegression(
    C=2.0,
    penalty='l2',
    solver='liblinear',
    random_state=42,
    max_iter=1000
)

xgb_model = XGBClassifier(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 集成模型
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lr', lr_model),
        ('rf', rf_model)
    ],
    voting='soft',
    n_jobs=-1
)

# 交叉验证评估
aucs = []
for train_idx, val_idx in skf.split(X_train_combined, y_train):
    X_train_fold, X_val_fold = X_train_combined[train_idx], X_train_combined[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    ensemble_model.fit(X_train_fold, y_train_fold)
    y_val_pred = ensemble_model.predict_proba(X_val_fold)[:, 1]
    aucs.append(roc_auc_score(y_val_fold, y_val_pred))

print(f"Ensemble model CV AUC: {np.mean(aucs):.6f}")

# 训练最终模型
print("Training final ensemble model...")
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
