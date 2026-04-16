import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 极简文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    return text

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 预处理训练数据
print("Preprocessing training data...")
train_texts = [preprocess_text(text) for text in train_df['review']]

# 预处理测试数据
print("Preprocessing test data...")
test_texts = [preprocess_text(text) for text in test_df['review']]

# 生成TF-IDF特征（限制特征数量）
print("Generating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=2000,  # 限制特征数量
    ngram_range=(1, 1),  # 只使用1-gram
    stop_words='english'
)

train_features = tfidf.fit_transform(train_texts)
test_features = tfidf.transform(test_texts)

# 准备训练数据
y_train = train_df['sentiment'].values

# 训练逻辑回归模型
print("Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=500,
    random_state=42
)

model.fit(train_features, y_train)

# 在测试集上预测
print("Predicting on test data...")
test_pred = model.predict_proba(test_features)[:, 1]

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': (test_pred > 0.5).astype(int)
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")