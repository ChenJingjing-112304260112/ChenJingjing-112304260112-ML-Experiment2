import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords

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
train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')

# 读取未标记数据
unlabeled_train_data = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t')

# 读取测试数据
test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

# 预处理训练数据（分类）
train_data['clean_words_classification'] = train_data['review'].apply(preprocess_text_classification)

# 预处理训练数据（Word2Vec）
train_data['clean_words_word2vec'] = train_data['review'].apply(preprocess_text_word2vec)

# 预处理未标记数据（Word2Vec）
unlabeled_train_data['clean_words_word2vec'] = unlabeled_train_data['review'].apply(preprocess_text_word2vec)

# 预处理测试数据（分类）
test_data['clean_words_classification'] = test_data['review'].apply(preprocess_text_classification)

# 预处理测试数据（Word2Vec）
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

# 训练Logistic Regression模型
print("Training Logistic Regression model...")
model = LogisticRegression(
    C=2.0,
    penalty='l2',
    solver='liblinear',
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

# 预测测试数据
print("Predicting test data...")
y_pred = model.predict(X_test)

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
from sklearn.model_selection import train_test_split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

val_model = LogisticRegression(
    C=2.0,
    penalty='l2',
    solver='liblinear',
    random_state=42,
    max_iter=1000
)

val_model.fit(X_train_split, y_train_split)
y_val_pred = val_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC score: {val_auc}")
