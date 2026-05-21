import pandas as pd
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from data_loader import get_dataloader
from models import MF, NCF

def train_and_predict(model_class, model_name, train_path, test_path,
                      num_users, num_items, embedding_dim=32, epochs=10, lr=0.001):
    # 1. 加载训练集
    # 2. 创建模型、损失函数、优化器
    # 3. 训练循环
    # 4. 对测试集预测，保存 pred_*.csv (user_id, item_id, true_rating, pred_rating)
    ...

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # 调用示例
    # train_and_predict(MF, "mf_dim32", "data/processed/train.csv", "data/processed/test.csv",
    #                   num_users=943, num_items=1682)