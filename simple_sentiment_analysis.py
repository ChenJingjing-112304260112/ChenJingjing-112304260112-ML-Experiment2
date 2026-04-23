import pandas as pd
import re

# 简单的情感分析函数
def simple_sentiment_analysis(text):
    # 预处理文本
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 定义情感词
    positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'awesome', 'love', 'like', 'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'terrific', 'enjoyable', 'entertaining', 'impressive', 'beautiful']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'poor', 'boring', 'dreadful', 'pathetic', 'lousy', 'horrendous', 'disgusting', 'hate', 'dislike', 'worse']
    
    # 计算情感分数
    words = text.split()
    score = 0
    for word in words:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    
    # 根据分数判断情感
    return 1 if score > 0 else 0

# 读取测试数据
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 生成预测
predictions = []
for text in test_df['review']:
    predictions.append(simple_sentiment_analysis(text))

# 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': predictions
})

# 保存提交文件
submission_df.to_csv('submission.csv', index=False)
print("Submission file generated successfully!")