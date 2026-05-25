"""
数据预处理与划分脚本
读取 data/raw/ml-100k/u.data，打乱并划分 80% 训练 / 20% 测试，保存到 data/processed/
"""
import pandas as pd
import os

print("正在读取 data/raw/ml-100k/u.data ...")
df = pd.read_csv(
    'data/raw/ml-100k/u.data',
    sep='\t',                      # 制表符分隔
    header=None,
    names=['user_id', 'item_id', 'rating', 'timestamp']
)
print(f"原始行数: {len(df)}")


df.drop(columns=['timestamp'], inplace=True)
print("已删除 timestamp 列")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("数据已随机打乱（seed=42）")

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"训练集行数: {len(train_df)}")
print(f"测试集行数: {len(test_df)}")

os.makedirs('data/processed', exist_ok=True)

train_df.to_csv('data/processed/train.csv', index=False)
print("已保存 data/processed/train.csv")

test_df.to_csv('data/processed/test.csv', index=False)
print("已保存 data/processed/test.csv")

print("\ntrain.csv 前 3 行 ---")
check = pd.read_csv('data/processed/train.csv', nrows=3)
print(check)