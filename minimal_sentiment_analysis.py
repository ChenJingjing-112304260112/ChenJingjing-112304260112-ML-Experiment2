import pandas as pd
import numpy as np

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

# 生成简单的预测（随机预测）
print("Generating predictions...")
np.random.seed(42)
test_pred = np.random.randint(0, 2, size=len(test_df))

# 生成提交文件
print("Generating submission file...")
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_pred
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")
print(f"Generated {len(submission_df)} predictions")
print("First few predictions:")
print(submission_df.head())