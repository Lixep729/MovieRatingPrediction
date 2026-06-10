import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mindspore as ms
import mindspore.ops as ops

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def read_pred_file(filepath):
    """读取预测文件，自动跳过可能存在的表头行"""
    df = pd.read_csv(filepath, header=None, dtype=str) 
    first_val = df.iloc[0, 0]
    try:
        float(first_val)
    except ValueError:
        df = df.iloc[1:].reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def cal_metrics(yt, yp):
    if len(yt) == 0:
        return 0.0, 0.0
    t_yt = ms.Tensor(yt.values if isinstance(yt, pd.Series) else yt, ms.float32)
    t_yp = ms.Tensor(yp.values if isinstance(yp, pd.Series) else yp, ms.float32)
    mae_val = ops.mean(ops.abs(t_yt - t_yp))
    rmse_val = ops.sqrt(ops.mean(ops.square(t_yt - t_yp)))
    return float(mae_val.asnumpy()), float(rmse_val.asnumpy())

# 1. 全模型性能对比
def eval_all(data_dir="outputs", test_path="data/processed/test.csv"):
    test_df = pd.read_csv(test_path)
    true_all = test_df['rating'].values

    all_files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    files = [f for f in all_files if "_test.csv" not in f]
    res_list = []

    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = read_pred_file(f)

        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            yt = true_all
            yp = df['yp']
        elif df.shape[1] >= 4:
            df.columns = ['uid', 'iid', 'yt', 'yp'] + [f'extra{i}' for i in range(df.shape[1]-4)]
            yt = df['yt']
            yp = df['yp']
        else:
            print(f"跳过 {m_name}：列数异常 ({df.shape[1]})")
            continue

        mae, rmse = cal_metrics(yt, yp)
        res_list.append({"Model": m_name, "MAE": mae, "RMSE": rmse})

    res_df = pd.DataFrame(res_list)
    if not res_df.empty:
        res_df = res_df.sort_values(by="RMSE").reset_index(drop=True)
        res_df.to_csv(os.path.join(data_dir, "model_comparison.csv"), index=False)
        print("\n=== 模型性能对比 ===")
        print(res_df)
    return res_df

# 2. 按用户活跃度分桶 RMSE 图
def plot_user_grp(data_dir, train_file, test_path="data/processed/test.csv"):
    tr_df = pd.read_csv(train_file)
    tr_df.columns = ['uid', 'iid', 'r']
    u_counts = tr_df['uid'].value_counts()

    test_df = pd.read_csv(test_path)
    true_all = test_df['rating'].values

    all_files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    files = [f for f in all_files if "_test.csv" not in f]
    plot_data = []

    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = read_pred_file(f)

        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            df['yt'] = true_all
        elif df.shape[1] >= 4:
            df.columns = ['uid', 'iid', 'yt', 'yp'] + [f'extra{i}' for i in range(df.shape[1]-4)]
        else:
            continue

        df['cnt'] = df['uid'].map(u_counts).fillna(0).astype(int)

        def get_grp(c):
            if c <= 20: return "冷启动(≤20)"
            elif c <= 100: return "中活跃(21-100)"
            else: return "高活跃(>100)"
        df['grp'] = df['cnt'].apply(get_grp)

        for grp_name, sub_df in df.groupby('grp', observed=False):
            _, rmse = cal_metrics(sub_df['yt'], sub_df['yp'])
            plot_data.append({"Model": m_name, "User Group": grp_name, "RMSE": rmse})

    p_df = pd.DataFrame(plot_data)
    if not p_df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=p_df, x="User Group", y="RMSE", hue="Model")
        plt.title("不同活跃度用户的 RMSE 对比")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "rmse_user.png"))
        plt.close()

# 3. 按电影热度分桶 RMSE 图
def plot_item_grp(data_dir, train_file, test_path="data/processed/test.csv"):
    tr_df = pd.read_csv(train_file)
    tr_df.columns = ['uid', 'iid', 'r']
    i_counts = tr_df['iid'].value_counts()

    test_df = pd.read_csv(test_path)
    true_all = test_df['rating'].values

    all_files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    files = [f for f in all_files if "_test.csv" not in f]
    plot_data = []

    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = read_pred_file(f)

        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            df['yt'] = true_all
        elif df.shape[1] >= 4:
            df.columns = ['uid', 'iid', 'yt', 'yp'] + [f'extra{i}' for i in range(df.shape[1]-4)]
        else:
            continue

        df['cnt'] = df['iid'].map(i_counts).fillna(0).astype(int)

        def get_item_grp(c):
            if c <= 10: return "冷门电影"
            elif c <= 50: return "普通电影"
            else: return "热门电影"
        df['grp'] = df['cnt'].apply(get_item_grp)

        for grp_name, sub_df in df.groupby('grp', observed=False):
            _, rmse = cal_metrics(sub_df['yt'], sub_df['yp'])
            plot_data.append({"Model": m_name, "Item Group": grp_name, "RMSE": rmse})

    p_df = pd.DataFrame(plot_data)
    if not p_df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=p_df, x="Item Group", y="RMSE", hue="Model")
        plt.title("不同热度电影的 RMSE 对比")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "rmse_item.png"))
        plt.close()

# 4. 残差分布图
def plot_err_dist(data_dir, test_path='data/processed/test.csv'):
    test_df = pd.read_csv(test_path)
    true_all = test_df['rating'].values

    left_models = [
        {'file': 'pred_global_mean.csv', 'label': 'Global Mean',    'color': '#F4A582', 'lw': 1.0},
        {'file': 'pred_user_avg.csv',   'label': 'User Average',   'color': '#E8833A', 'lw': 1.2},
        {'file': 'pred_item_knn.csv',   'label': 'Item-KNN',       'color': '#CA0020', 'lw': 1.5},
    ]
    right_models = [
        {'file': 'pred_mf_dim64.csv',   'label': 'MF (dim=64)',     'color': '#0571B0', 'lw': 2.0},
        {'file': 'pred_ncf_layer3.csv', 'label': 'NCF [128,64,32]', 'color': '#008837', 'lw': 2.0},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for ax, model_list in zip(axes, [left_models, right_models]):
        for mdl in model_list:
            f = os.path.join(data_dir, mdl['file'])
            if not os.path.exists(f):
                print(f"文件缺失: {mdl['file']}")
                continue
            df = read_pred_file(f)
            if df.shape[1] == 3:
                df.columns = ['uid', 'iid', 'yp']
                yt = true_all
                yp = df['yp']
            else:
                df.columns = ['uid', 'iid', 'yt', 'yp'] + [f'extra{i}' for i in range(df.shape[1]-4)]
                yt = df['yt']
                yp = df['yp']
            err = yt - yp
            sns.histplot(err, kde=True, label=mdl['label'], stat="density",
                         bins=40, color=mdl['color'], alpha=0.35, linewidth=mdl['lw'], ax=ax)
        ax.axvline(x=0, color='#333333', linestyle='--', linewidth=1.2)
        ax.set_title('Baseline Models' if ax == axes[0] else 'Advanced Models', fontsize=13, fontweight='bold')
        ax.set_xlabel('Prediction Error', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9, frameon=True)
        ax.grid(axis='y', alpha=0.2)

    plt.suptitle('Prediction Error Distribution Comparison', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = os.path.join(data_dir, 'err_dist_compare.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"图片已保存至 {output_path}")

# 5. MF 维度 - RMSE 趋势图
def plot_mf_trend(data_dir):
    files = glob.glob(os.path.join(data_dir, "pred_mf_dim*.csv"))
    files = [f for f in files if "_test.csv" not in f]
    if not files:
        return

    res = []
    for f in files:
        try:
            dim_str = os.path.basename(f).replace("pred_mf_dim", "").replace(".csv", "")
            dim = int(dim_str)
            df = read_pred_file(f)
            if df.shape[1] >= 4:
                df.columns = ['uid', 'iid', 'yt', 'yp'] + [f'extra{i}' for i in range(df.shape[1]-4)]
            else:
                continue  
            _, rmse = cal_metrics(df['yt'], df['yp'])
            res.append({'Dim': dim, 'RMSE': rmse})
        except ValueError:
            continue

    res_df = pd.DataFrame(res).sort_values('Dim')
    if not res_df.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(res_df['Dim'], res_df['RMSE'], marker='o', linestyle='-', color='b', linewidth=2)
        plt.title("MF 模型 Embedding 维度对 RMSE 的影响趋势")
        plt.xlabel("Embedding 维度 (Dim)")
        plt.ylabel("RMSE")
        plt.xticks(res_df['Dim'])
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "mf_dim_trend.png"))
        plt.close()


# 6. 数据稀疏性专题分析
def calc_sparse_gap(data_dir, train_file, test_path="data/processed/test.csv"):
    tr_df = pd.read_csv(train_file)
    tr_df.columns = ['uid', 'iid', 'r']
    u_counts = tr_df['uid'].value_counts()
    u_counts.index = u_counts.index.astype(int)

    test_df = pd.read_csv(test_path)
    true_all = test_df['rating'].values

    all_files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    files = [f for f in all_files if "_test.csv" not in f]
    gap_data = []

    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = read_pred_file(f)

        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            df['yt'] = true_all
        elif df.shape[1] >= 4:
            df.columns = ['uid', 'iid', 'yt', 'yp'] + [f'extra{i}' for i in range(df.shape[1]-4)]
        else:
            continue

        df['uid'] = pd.to_numeric(df['uid'], errors='coerce')
        df = df.dropna(subset=['uid'])
        df['uid'] = df['uid'].astype(int)

        df['cnt'] = df['uid'].map(u_counts).fillna(0).astype(int)

        _, rmse_all = cal_metrics(df['yt'], df['yp'])
        df_cold = df[df['cnt'] <= 20]

        if len(df_cold) > 0 and rmse_all > 0:
            _, rmse_cold = cal_metrics(df_cold['yt'], df_cold['yp'])
            up_ratio = (rmse_cold - rmse_all) / rmse_all * 100
            gap_data.append({
                "Model": m_name,
                "整体 RMSE": round(rmse_all, 4),
                "冷启动 RMSE": round(rmse_cold, 4),
                "恶化比例(%)": round(up_ratio, 2)
            })

    if gap_data:
        gap_df = pd.DataFrame(gap_data).sort_values("恶化比例(%)")
        print("\n=== 数据稀疏性专题分析 ===")
        print(gap_df)
        gap_df.to_csv(os.path.join(data_dir, "sparsity_analysis.csv"), index=False)

if __name__ == "__main__":
    out_dir = "outputs"
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    print("开始运行全模型完整指标评估")
    eval_all(out_dir, test_path)
    plot_user_grp(out_dir, train_path, test_path)
    plot_item_grp(out_dir, train_path, test_path)
    plot_err_dist(out_dir, test_path)
    plot_mf_trend(out_dir)
    calc_sparse_gap(out_dir, train_path, test_path)
    print("\n所有可视化图表与数据分析已成功生成在outputs/目录下！")