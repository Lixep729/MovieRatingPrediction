import pandas as pd
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops
import mindspore.numpy as mnp

def predict_user_mean(train_df, test_df):
    # 关键修复：转换为 Python int
    n_users = int(max(train_df['user_id'].max(), test_df['user_id'].max()))

    train_users = Tensor(train_df['user_id'].values.astype(np.int32) - 1)
    train_ratings = Tensor(train_df['rating'].values.astype(np.float32))

    user_means = unsorted_segment_mean_custom(train_ratings, train_users, n_users)

    ones = ops.Ones()(train_ratings.shape, ms.float32)
    user_counts = ops.unsorted_segment_sum(ones, train_users, n_users)  # 现在 n_users 是 int

    test_users = Tensor(test_df['user_id'].values.astype(np.int32) - 1)
    preds = ops.gather(user_means, test_users, 0)
    counts = ops.gather(user_counts, test_users, 0)

    global_mean = train_ratings.mean()
    cold_mask = ops.equal(counts, 0)
    preds = ops.select(cold_mask,
                       ops.fill(ms.float32, preds.shape, global_mean),
                       preds)

    pred_df = test_df[['user_id', 'item_id']].copy()
    pred_df['pred_rating'] = preds.asnumpy()
    return pred_df

def unsorted_segment_mean_custom(data, segment_ids, num_segments):
    """自定义的 unsorted_segment_mean 实现"""
    num_segments = int(num_segments) 
    # 1. 计算各segment的评分总和
    segment_sum = ops.unsorted_segment_sum(data, segment_ids, num_segments)
    
    # 2. 计算各segment的元素个数
    ones = ops.Ones()(data.shape, ms.float32)
    segment_counts = ops.unsorted_segment_sum(ones, segment_ids, num_segments)
    
    # 3. 处理空segment，避免除以零
    segment_counts = ops.maximum(segment_counts, Tensor(1.0, ms.float32))
    
    # 4. 计算平均值
    return segment_sum / segment_counts

def predict_user_mean(train_df, test_df):
    n_users = int(max(train_df['user_id'].max(), test_df['user_id'].max()))

    train_users = Tensor(train_df['user_id'].values.astype(np.int32) - 1)
    train_ratings = Tensor(train_df['rating'].values.astype(np.float32))

    # 调用自定义函数，原代码其他部分无需修改
    user_means = unsorted_segment_mean_custom(train_ratings, train_users, n_users)
    user_counts = ops.unsorted_segment_sum(ops.Ones()(train_ratings.shape, ms.float32), train_users, n_users)

    # 测试用户索引
    test_users = Tensor(test_df['user_id'].values.astype(np.int32) - 1)
    preds = ops.Gather()(user_means, test_users, 0)      # 查表得到预测值
    counts = ops.Gather()(user_counts, test_users, 0)    # 查表得到评分次数

    # 冷启动处理：counts==0 的用户用全局平均替代
    global_mean = train_ratings.mean()
    cold_mask = ops.Equal()(counts, 0)
    preds = ops.Select()(cold_mask,
                         ops.Fill()(ms.float32, preds.shape, global_mean),
                         preds)

    pred_df = test_df[['user_id', 'item_id']].copy()
    pred_df['pred_rating'] = preds.asnumpy()
    return pred_df

def predict_item_knn(train_df, test_df, K=20):
    n_users = int(max(train_df['user_id'].max(), test_df['user_id'].max()))
    n_items = int(max(train_df['item_id'].max(), test_df['item_id'].max()))

    user_idx = Tensor(train_df['user_id'].values.astype(np.int32) - 1)
    item_idx = Tensor(train_df['item_id'].values.astype(np.int32) - 1)
    ratings = Tensor(train_df['rating'].values.astype(np.float32))

    indices = ops.Stack(axis=1)([user_idx, item_idx])
    
    # 修复：shape 元素必须是 Python int
    shape = (int(n_users), int(n_items))
    R = ops.ScatterNd()(indices, ratings, shape)

    # ---- 3. 物品相似度矩阵（余弦） ----
    # 转置得到 (n_items, n_users)
    item_vecs = ops.Transpose()(R, (1, 0))

    # 按行计算 L2 范数并归一化
    norms = ops.L2Normalize(axis=1)(item_vecs)   # 或者手动：eps=1e-8, item_vecs / sqrt(sum(item_vecs^2, dim=1))
    # 注意：L2Normalize 已经做了归一化，但需要确认 MindSpore 版本是否支持。保险起见用以下方式：
    # l2 = ops.Sqrt()(ops.ReduceSum()(item_vecs * item_vecs, 1, keepdims=True))
    # l2 = ops.Maximum()(l2, Tensor(1e-8, ms.float32))
    # item_vecs = item_vecs / l2
    # 这里我们使用 L2Normalize
    item_vecs = ops.L2Normalize(axis=1)(item_vecs)

    # 余弦相似度 = 归一化向量点乘
    sim_matrix = ops.MatMul()(item_vecs, ops.Transpose()(item_vecs, (1, 0)))

    # ---- 4. 为每个物品找出 K 个最相似邻居（排除自身） ----
    # 自身相似度设为极小值
    eye = ops.Eye()(n_items, n_items, ms.float32)
    sim_matrix = sim_matrix - eye * 1e9

    # 取 top‑k
    topk_values, topk_indices = ops.TopK()(sim_matrix, K)  # 默认 dim=-1，即每一行

    # ---- 5. 逐样本预测 ----
    test_users = Tensor(test_df['user_id'].values.astype(np.int32) - 1)
    test_items = Tensor(test_df['item_id'].values.astype(np.int32) - 1)

    global_mean = ratings.mean().asnumpy().item()

    pred_list = []
    for i in range(len(test_df)):
        u = test_users[i]
        item = test_items[i]

        neigh_idx = topk_indices[item]   # (K,)
        neigh_sim = topk_values[item]    # (K,)

        # 用户 u 对 K 个邻居的评分
        # GatherNd 需要二维索引：[[u, n1], [u, n2], ...]
        gather_indices = ops.Stack(axis=1)([
            ops.Fill()(ms.int32, (K,), u),
            ops.Cast()(neigh_idx, ms.int32)
        ])
        neigh_ratings = ops.GatherNd()(R, gather_indices)  # (K,)

        # 只考虑有评分的邻居（评分>0）
        valid = neigh_ratings > 0
        valid_ratings = ops.MaskedSelect()(neigh_ratings, valid)
        valid_sims = ops.MaskedSelect()(neigh_sim, valid)

        if valid_ratings.shape[0] > 0:
            weighted_sum = ops.ReduceSum()(valid_ratings * valid_sims)
            norm = ops.ReduceSum()(ops.Abs()(valid_sims))
            if norm > 0:
                pred = (weighted_sum / norm).asnumpy().item()
            else:
                pred = global_mean
        else:
            pred = global_mean

        pred_list.append(pred)

    pred_df = test_df[['user_id', 'item_id']].copy()
    pred_df['pred_rating'] = pred_list
    return pred_df

def save_predictions(pred_df, model_name, output_dir="outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/pred_{model_name}.csv"
    pred_df.to_csv(path, index=False, header=False)

if __name__ == '__main__':
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')

    pred_a = predict_user_mean(train, test)
    save_predictions(pred_a, 'global_avg')
    print('模型A 已保存')

    pred_b = predict_user_mean(train, test)
    save_predictions(pred_b, 'user_avg')
    print('模型B 已保存')

    pred_c = predict_item_knn(train, test)
    save_predictions(pred_c, 'item_knn')
    print('模型C 已保存')