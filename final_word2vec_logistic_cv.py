import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
import os

# 下载必要的NLTK资源
nltk.download('stopwords', quiet=True)

# 文本预处理函数（用于分类）
def preprocess_text_classification(text):
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
    # 过滤停用词
    words = [word for word in words if word not in stop_words]
    return words

# 文本预处理函数（用于Word2Vec训练）
def preprocess_text_word2vec(text):
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 提取英文单词，保留带缩写的否定形式
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    # 分词
    words = text.split()
    # 训练Word2Vec时不删除停用词
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

# 预处理训练数据（分类）
print("Preprocessing training data for classification...")
train_data['clean_words_classification'] = train_data['review'].apply(preprocess_text_classification)

# 预处理训练数据（Word2Vec）
print("Preprocessing training data for Word2Vec...")
train_data['clean_words_word2vec'] = train_data['review'].apply(preprocess_text_word2vec)

# 预处理未标记数据（Word2Vec）
print("Preprocessing unlabeled training data for Word2Vec...")
unlabeled_train_data['clean_words_word2vec'] = unlabeled_train_data['review'].apply(preprocess_text_word2vec)

# 预处理测试数据（分类）
print("Preprocessing test data for classification...")
test_data['clean_words_classification'] = test_data['review'].apply(preprocess_text_classification)

# 预处理测试数据（Word2Vec）
print("Preprocessing test data for Word2Vec...")
test_data['clean_words_word2vec'] = test_data['review'].apply(preprocess_text_word2vec)

# 训练Word2Vec模型
print("Training Word2Vec model...")
all_words = train_data['clean_words_word2vec'].tolist() + \
            unlabeled_train_data['clean_words_word2vec'].tolist() + \
            test_data['clean_words_word2vec'].tolist()

word2vec_model = Word2Vec(
    sentences=all_words,
    vector_size=300,
    window=10,
    min_count=40,
    workers=4,
    sg=1,  # Skip-gram
    hs=1,  # Hierarchical Softmax
    sample=0.001,  # 高频词下采样
    epochs=10
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
X_train = np.array([get_mean_embedding(words, word2vec_model, 300) for words in train_data['clean_words_classification']])
y_train = train_data['sentiment']
X_test = np.array([get_mean_embedding(words, word2vec_model, 300) for words in test_data['clean_words_classification']])

# 分层5折交叉验证评估不同分类器
print("Performing 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 评估Logistic Regression
lr_aucs = []
for train_idx, val_idx in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    lr_model = LogisticRegression(
        C=2.0,
        penalty='l2',
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    lr_model.fit(X_train_fold, y_train_fold)
    y_val_pred = lr_model.predict_proba(X_val_fold)[:, 1]
    lr_aucs.append(roc_auc_score(y_val_fold, y_val_pred))

print(f"Logistic Regression (C=2.0) CV AUC: {np.mean(lr_aucs):.6f}")

# 评估Random Forest
rf_aucs = []
for train_idx, val_idx in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    rf_model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_fold, y_train_fold)
    y_val_pred = rf_model.predict_proba(X_val_fold)[:, 1]
    rf_aucs.append(roc_auc_score(y_val_fold, y_val_pred))

print(f"Random Forest (400 estimators) CV AUC: {np.mean(rf_aucs):.6f}")

# 评估LinearSVC
lsvc_aucs = []
for train_idx, val_idx in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    lsvc_model = LinearSVC(
        C=1.0,
        random_state=42,
        max_iter=10000
    )
    lsvc_model.fit(X_train_fold, y_train_fold)
    # LinearSVC没有predict_proba方法，使用decision_function
    y_val_pred = lsvc_model.decision_function(X_val_fold)
    lsvc_aucs.append(roc_auc_score(y_val_fold, y_val_pred))

print(f"LinearSVC (C=1.0) CV AUC: {np.mean(lsvc_aucs):.6f}")

# 选取最优模型在全训练集上训练
print("Training optimal model on full training set...")
optimal_model = LogisticRegression(
    C=2.0,
    penalty='l2',
    solver='liblinear',
    random_state=42,
    max_iter=1000
)

optimal_model.fit(X_train, y_train)

# 预测测试数据
print("Predicting test data...")
y_pred = optimal_model.predict(X_test)

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
