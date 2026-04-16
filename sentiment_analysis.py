import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

# 下载必要的NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 去除标点和特殊符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 读取无标签数据（用于训练Word2Vec）
print("Reading unlabeled data...")
unlabeled_df = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t', quoting=3)

# 预处理所有文本数据
print("Preprocessing text data...")
train_tokens = []
for text in train_df['review']:
    train_tokens.append(preprocess_text(text))

test_tokens = []
for text in test_df['review']:
    test_tokens.append(preprocess_text(text))

unlabeled_tokens = []
for text in unlabeled_df['review']:
    unlabeled_tokens.append(preprocess_text(text))

# 合并所有文本用于Word2Vec训练
all_tokens = train_tokens + test_tokens + unlabeled_tokens

# 训练Word2Vec模型
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    sentences=all_tokens,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=10
)

# 句子向量表示（平均词向量）
def get_sentence_vector(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 生成训练集和测试集的向量
print("Generating text vectors...")
train_vectors = []
for tokens in train_tokens:
    train_vectors.append(get_sentence_vector(tokens, word2vec_model))
train_vectors = np.array(train_vectors)

test_vectors = []
for tokens in test_tokens:
    test_vectors.append(get_sentence_vector(tokens, word2vec_model))
test_vectors = np.array(test_vectors)

# 准备训练数据
y_train = train_df['sentiment'].values

# 训练分类模型
print("Training classification model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(train_vectors, y_train)

# 在测试集上预测
print("Predicting on test data...")
test_pred = model.predict(test_vectors)

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_pred
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

# 保存Word2Vec模型
word2vec_model.save('word2vec_model.model')
print("Word2Vec model saved: word2vec_model.model")