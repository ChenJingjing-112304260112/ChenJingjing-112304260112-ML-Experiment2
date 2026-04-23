import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords

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
    # 移除标点符号，但保留必要的空格
    text = re.sub(r"[^a-zA-Z\s]", ' ', text)
    # 分词（简单分词，保留短语）
    words = text.split()
    # 移除停用词（保留否定词）
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor'}
    stop_words = stop_words - negation_words
    # 过滤停用词
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# 读取训练数据
print("Reading training data...")
train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')

# 读取测试数据
print("Reading test data...")
test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

# 预处理训练数据
print("Preprocessing training data...")
train_data['clean_review'] = train_data['review'].apply(preprocess_text)

# 预处理测试数据
print("Preprocessing test data...")
test_data['clean_review'] = test_data['review'].apply(preprocess_text)

# 提取TF-IDF特征（使用短语模式）
print("Extracting TF-IDF features with phrase patterns...")
vectorizer = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1, 3),  # 短语模式：1-3gram
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    use_idf=True
)

X_train = vectorizer.fit_transform(train_data['clean_review'])
y_train = train_data['sentiment']
X_test = vectorizer.transform(test_data['clean_review'])

# 训练逻辑回归模型
print("Training Logistic Regression model...")
model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='saga',
    random_state=42,
    max_iter=2000,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 验证模型性能
print("Evaluating model performance...")
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

val_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='saga',
    random_state=42,
    max_iter=2000,
    n_jobs=-1
)

val_model.fit(X_train_split, y_train_split)
y_val_pred = val_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC score: {val_auc:.6f}")

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
