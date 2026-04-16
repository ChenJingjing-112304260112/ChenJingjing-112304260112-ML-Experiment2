import pandas as pd
import re

# 简单的文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    return text

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 计算训练数据的平均长度
print("Calculating average review length...")
train_df['review_length'] = train_df['review'].apply(len)
average_length = train_df['review_length'].mean()
print(f"Average review length: {average_length}")

# 基于文本长度的简单预测：长评论更可能是正面的
print("Predicting on test data...")
test_df['review_length'] = test_df['review'].apply(len)
test_df['predicted'] = (test_df['review_length'] > average_length).astype(int)

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': test_df['predicted']
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")

print("Task completed successfully!")