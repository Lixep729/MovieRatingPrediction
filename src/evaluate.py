import pandas as pd
import numpy as np
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor
import glob
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred):
    y_true = Tensor(y_true, ms.float32)
    y_pred = Tensor(y_pred, ms.float32)
    mae = ops.mean(ops.abs(y_true - y_pred))
    rmse = ops.sqrt(ops.mean((y_true - y_pred) ** 2))
    return mae.asnumpy().item(), rmse.asnumpy().item()

def load_prediction(filepath):
    return pd.read_csv(filepath, header=None,
                       names=['user_id', 'item_id', 'true_rating', 'pred_rating'])

def evaluate_all_models(pred_dir="outputs"):
    pred_files = glob.glob(f"{pred_dir}/pred_*.csv")
    results = []
    for f in pred_files:
        model_name = f.replace("\\", "/").split("/")[-1].replace("pred_", "").replace(".csv", "")
        df = load_prediction(f)
        mae, rmse = compute_metrics(df['true_rating'].values, df['pred_rating'].values)
        results.append({"Model": model_name, "MAE": mae, "RMSE": rmse})
    result_df = pd.DataFrame(results).sort_values("RMSE")
    result_df.to_csv(f"{pred_dir}/model_comparison.csv", index=False)
    return result_df

def plot_error_by_user_group(pred_dir="outputs", train_path="data/processed/train.csv"):
    # 按用户评分数分组绘制误差柱状图
    ...

def plot_error_distribution(pred_dir="outputs"):
    # 绘制预测偏差直方图
    ...