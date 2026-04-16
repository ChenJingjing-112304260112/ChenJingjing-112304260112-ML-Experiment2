import pandas as pd
import re
from gensim.models import Word2Vec

# 简单的文本预处理函数
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()

# 读取少量训练数据
print("Reading small sample of training data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3, nrows=1000)

# 预处理文本
print("Preprocessing text...")
tokens = [preprocess_text(text) for text in train_df['review']]

print(f"Processed {len(tokens)} reviews")
print(f"First review tokens: {tokens[0][:10]}...")

# 训练Word2Vec模型
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    sentences=tokens,
    vector_size=50,
    window=5,
    min_count=2,
    workers=2,
    epochs=5
)

# 保存模型
word2vec_model.save('test_word2vec.model')
print("Word2Vec model saved: test_word2vec.model")

# 测试模型
print("Testing Word2Vec model...")
if 'good' in word2vec_model.wv:
    print("Word 'good' found in vocabulary")
    similar_words = word2vec_model.wv.most_similar('good', topn=5)
    print(f"Words similar to 'good': {similar_words}")
else:
    print("Word 'good' not found in vocabulary")

print("Test completed successfully!")