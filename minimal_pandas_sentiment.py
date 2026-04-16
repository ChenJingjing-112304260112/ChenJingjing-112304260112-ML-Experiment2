import pandas as pd
import numpy as np
import re

# 简单的文本预处理函数
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
train_df['clean_review'] = train_df['review'].apply(preprocess_text)

# 预处理测试数据
print("Preprocessing test data...")
test_df['clean_review'] = test_df['review'].apply(preprocess_text)

# 简单的特征提取：计算正面词和负面词的数量
print("Extracting features...")

# 定义正面和负面词
positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'awesome', 'love', 'like', 'best']
negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'poor', 'boring', 'hate', 'dislike']

# 计算正面词数量
def count_positive(text):
    return sum(1 for word in positive_words if word in text)

# 计算负面词数量
def count_negative(text):
    return sum(1 for word in negative_words if word in text)

# 提取特征
train_df['positive_count'] = train_df['clean_review'].apply(count_positive)
train_df['negative_count'] = train_df['clean_review'].apply(count_negative)
train_df['sentiment_score'] = train_df['positive_count'] - train_df['negative_count']

test_df['positive_count'] = test_df['clean_review'].apply(count_positive)
test_df['negative_count'] = test_df['clean_review'].apply(count_negative)
test_df['sentiment_score'] = test_df['positive_count'] - test_df['negative_count']

# 简单的阈值分类
print("Training simple threshold model...")

# 找到最佳阈值
best_threshold = 0
best_accuracy = 0

for threshold in range(-5, 6):
    correct = 0
    for i, score in enumerate(train_df['sentiment_score']):
        predicted = 1 if score > threshold else 0
        if predicted == train_df['sentiment'].iloc[i]:
            correct += 1
    accuracy = correct / len(train_df)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, accuracy: {best_accuracy:.4f}")

# 在测试集上预测
print("Predicting on test data...")
test_df['predicted'] = (test_df['sentiment_score'] > best_threshold).astype(int)

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': test_df['predicted']
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")