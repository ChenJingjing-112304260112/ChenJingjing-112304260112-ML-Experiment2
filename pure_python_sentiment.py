import pandas as pd
import re
import math

# 简单的文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转小写
    text = text.lower()
    # 保留否定词的特殊处理
    text = re.sub(r"n't", " not", text)
    # 去除标点和特殊符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    return text.split()

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 预处理训练数据
print("Preprocessing training data...")
train_tokens = []
for text in train_df['review']:
    train_tokens.append(preprocess_text(text))

# 预处理测试数据
print("Preprocessing test data...")
test_tokens = []
for text in test_df['review']:
    test_tokens.append(preprocess_text(text))

# 构建情感词典
print("Building sentiment lexicon...")

# 简单的情感词典（正面和负面词）
positive_words = set(['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'awesome', 'love', 'like', 'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'terrific', 'enjoyable', 'entertaining', 'impressive', 'beautiful'])
negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'poor', 'boring', 'dreadful', 'pathetic', 'lousy', 'horrendous', 'disgusting', 'awful', 'terrible', 'negative', 'hate', 'dislike', 'worse'])

# 计算情感分数
def calculate_sentiment_score(tokens):
    score = 0
    for token in tokens:
        if token in positive_words:
            score += 1
        elif token in negative_words:
            score -= 1
    return score

# 训练简单的阈值模型
print("Training simple threshold model...")
sentiment_scores = []
for tokens in train_tokens:
    score = calculate_sentiment_score(tokens)
    sentiment_scores.append(score)

# 找到最佳阈值
best_threshold = 0
best_accuracy = 0

for threshold in range(-10, 11):
    correct = 0
    for i, score in enumerate(sentiment_scores):
        predicted = 1 if score > threshold else 0
        if predicted == train_df['sentiment'].iloc[i]:
            correct += 1
    accuracy = correct / len(sentiment_scores)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, accuracy: {best_accuracy:.4f}")

# 在测试集上预测
print("Predicting on test data...")
test_predictions = []
for tokens in test_tokens:
    score = calculate_sentiment_score(tokens)
    predicted = 1 if score > best_threshold else 0
    test_predictions.append(predicted)

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': test_predictions
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")