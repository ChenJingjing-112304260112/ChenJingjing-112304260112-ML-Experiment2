import pandas as pd

# 读取submission.csv文件
df = pd.read_csv('submission.csv')

# 打印行数
print(f"文件行数: {len(df)}")
print(f"列名: {list(df.columns)}")
print(f"前5行数据:")
print(df.head())
