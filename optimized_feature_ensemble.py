import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 下载必要的NLTK资源
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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
    return ' '.join(words)

# 读取训练数据
train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')

# 预处理训练数据
train_data['clean_review'] = train_data['review'].apply(preprocess_text)

# 读取测试数据
test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

# 预处理测试数据
test_data['clean_review'] = test_data['review'].apply(preprocess_text)

# 提取TF-IDF特征（优化参数）
vectorizer = TfidfVectorizer(
    max_features=30000,  # 增加特征数量
    ngram_range=(1, 3),  # 增加ngram范围
    min_df=2,  # 最小文档频率
    max_df=0.9  # 最大文档频率
)

X_train = vectorizer.fit_transform(train_data['clean_review'])
y_train = train_data['sentiment']
X_test = vectorizer.transform(test_data['clean_review'])

# 定义多个分类模型
model1 = XGBClassifier(
    n_estimators=1500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model2 = RandomForestClassifier(
    n_estimators=1000,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model3 = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    random_state=42,
    max_iter=1000
)

# 创建集成模型
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', model1),
        ('rf', model2),
        ('lr', model3)
    ],
    voting='soft',  # 使用软投票
    n_jobs=-1
)

# 训练集成模型
print("Training ensemble model...")
ensemble_model.fit(X_train, y_train)

# 预测测试数据
print("Predicting test data...")
y_pred = ensemble_model.predict(X_test)

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
