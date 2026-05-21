import pandas as pd
import numpy as np
import mindspore as ms
import mindspore.ops as ops

def predict_global_mean(train_df, test_df):
    # 返回 DataFrame: user_id, item_id, pred_rating
    ...

def predict_user_mean(train_df, test_df):
    # 用户平均分，冷启动回退全局均值
    ...

def predict_item_knn(train_df, test_df, k=20):
    # 基于物品的 KNN，使用 MindSpore 算子
    ...

def save_predictions(pred_df, model_name, output_dir="outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/pred_{model_name}.csv"
    pred_df.to_csv(path, index=False, header=False)