from src.data_loader import get_dataloader

# 加载训练集
train_loader = get_dataloader("data/processed/train.csv")

# 测试输出2个batch
for i, batch in enumerate(train_loader):
    print(f"第{i+1}个batch：")
    print(batch)
    if i >= 1:
        break