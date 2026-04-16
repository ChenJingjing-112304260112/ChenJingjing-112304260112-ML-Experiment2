import pandas as pd

# 读取生成的提交文件
df = pd.read_csv('submission.csv')

# 修复ID字段，去除多余的引号
df['id'] = df['id'].str.replace('"', '')

# 保存修复后的文件
df.to_csv('submission.csv', index=False)

print("Submission file fixed!")
print("First few rows:")
print(df.head())