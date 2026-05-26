# src/train.py
import os
import pandas as pd
import mindspore as ms
from mindspore import nn, ops
from models import MF, NCF
from data_loader import get_dataloader


def train_and_predict(
    model_class,          # 模型类，如 MF 或 NCF
    model_name: str,      # 用于命名输出文件，例如 "mf_dim32"
    train_path: str,      # 训练集 CSV 路径
    test_path: str,       # 测试集 CSV 路径
    num_users: int,       # 用户总数
    num_items: int,       # 物品总数
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 256,
    **model_kwargs        # 传递给模型的参数，如 embedding_dim=32
):
    
    # 1. 数据加载
    train_dataset = get_dataloader(train_path, batch_size=batch_size, shuffle=True)
    test_dataset = get_dataloader(test_path, batch_size=batch_size, shuffle=False)

    # 2. 模型、损失函数、优化器
    model = model_class(num_users, num_items,** model_kwargs)
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    # 3. 封装前向与梯度计算
    def forward_fn(data, label):
        user_ids, item_ids = data
        pred = model(user_ids, item_ids)
        loss = loss_fn(pred, label)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    # 4. 训练循环
    print(f"Start training: {model_name}")
    for epoch in range(epochs):
        model.set_train(True)
        total_loss = 0.0
        steps = 0
        for batch in train_dataset.create_tuple_iterator():
            user_ids, item_ids, ratings = batch
            loss = train_step((user_ids, item_ids), ratings)
            total_loss += float(loss.asnumpy())
            steps += 1
        avg_loss = total_loss / steps
        print(f" Epoch {epoch+1:2d}/{epochs}  Avg Loss: {avg_loss:.6f}")

    # 5. 测试集预测
    model.set_train(False)
    user_list, item_list, true_list, pred_list = [], [], [], []
    for batch in test_dataset.create_tuple_iterator():
        user_ids, item_ids, ratings = batch
        preds = model(user_ids, item_ids)
        user_list.extend(user_ids.asnumpy().flatten().tolist())
        item_list.extend(item_ids.asnumpy().flatten().tolist())
        true_list.extend(ratings.asnumpy().flatten().tolist())
        pred_list.extend(preds.asnumpy().flatten().tolist())

    # 6. 保存预测结果
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"pred_{model_name}.csv")
    result_df = pd.DataFrame({
        "user_id": user_list,
        "item_id": item_list,
        "true_rating": true_list,
        "pred_rating": pred_list
    })
    result_df["user_id"] = result_df["user_id"].astype(int)
    result_df["item_id"] = result_df["item_id"].astype(int)

    result_df.to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")
    return output_file


# ==================== 主程序测试入口 ====================
if __name__ == "__main__":
    TRAIN_PATH = "data/processed/train.csv"
    TEST_PATH = "data/processed/test.csv"

    NUM_USERS = 944
    NUM_ITEMS = 1683

    print("="*50)
    print("Running quick test for MF model (dim=16)")
    print("="*50)

    train_and_predict(
        model_class=MF,
        model_name="mf_dim16_test",
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        epochs=3,
        lr=0.001,
        batch_size=256,
        embedding_dim=16
    )

    print("\n" + "="*50)
    print("Running quick test for NCF model (layers=[64,32,16])")
    print("="*50)

    train_and_predict(
        model_class=NCF,
        model_name="ncf_layer3_test",
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        epochs=3,
        lr=0.001,
        batch_size=256,
        layers=[64, 32, 16]
    )

    print("\nAll tests completed. Check outputs/ for prediction files.")