import pandas as pd
import re

# 简单的情感分析函数，使用不同的方法
def simple_sentiment_analysis_different(text):
    # 预处理文本
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 定义更丰富的情感词
    positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'awesome', 'love', 'like', 'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'terrific', 'enjoyable', 'entertaining', 'impressive', 'beautiful', 'excellent', 'wonderful', 'fantastic', 'amazing', 'incredible', 'marvelous', 'splendid', 'magnificent', 'exceptional', 'remarkable']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'poor', 'boring', 'dreadful', 'pathetic', 'lousy', 'horrendous', 'disgusting', 'hate', 'dislike', 'worse', 'awful', 'terrible', 'negative', 'horrible', 'dreadful', 'appalling', 'abysmal', 'atrocious', 'awful', 'disastrous', 'dismal', 'horrendous', 'miserable', 'pathetic']
    
    # 计算情感分数，使用不同的加权方法
    words = text.split()
    score = 0
    for i, word in enumerate(words):
        if word in positive_words:
            # 给正面词更高的权重
            score += 1.5
        elif word in negative_words:
            # 给负面词更高的权重
            score -= 1.5
        # 考虑否定词的影响
        if i > 0 and words[i-1] in ['not', 'no', 'never', 'nothing', 'nowhere', 'noone', 'none', 'cannot']:
            # 否定词会反转前一个词的情感
            if word in positive_words:
                score -= 3.0
            elif word in negative_words:
                score += 3.0
    
    # 根据分数判断情感，使用不同的阈值
    return 1 if score > 0.5 else 0

# 读取测试数据
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 生成预测
predictions = []
for text in test_df['review']:
    predictions.append(simple_sentiment_analysis_different(text))

# 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'].str.replace('"', ''),
    'sentiment': predictions
})

# 保存提交文件
submission_df.to_csv('submission.csv', index=False)
print("Submission file generated successfully with different approach!")