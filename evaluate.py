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

# 1.计算指标，用MindSpore算子
def cal_metrics(yt, yp):
    if len(yt) == 0:  
        return 0.0, 0.0
    t_yt = ms.Tensor(yt, ms.float32)
    t_yp = ms.Tensor(yp, ms.float32)
    
    mae_val = ops.mean(ops.abs(t_yt - t_yp))
    rmse_val = ops.sqrt(ops.mean(ops.square(t_yt - t_yp)))
    
    return float(mae_val.asnumpy()), float(rmse_val.asnumpy())

# 2.自动评估outputs文件夹下的所有正式预测结果
def eval_all(data_dir="outputs"):
    all_files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    #过滤掉测试后缀和包含复杂参数列表的冗余文件
    files = [f for f in all_files if "_test.csv" not in f and "layers[" not in f]
    res_list = []
    
    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = pd.read_csv(f, header=None, names=['uid', 'iid', 'yt', 'yp'])
        
        mae, rmse = cal_metrics(df['yt'].values, df['yp'].values)
        res_list.append({"Model": m_name, "MAE": mae, "RMSE": rmse})
        
    res_df = pd.DataFrame(res_list)
    if not res_df.empty:
        res_df = res_df.sort_values(by="RMSE").reset_index(drop=True)
        res_df.to_csv(os.path.join(data_dir, "model_comparison.csv"), index=False)
        print("\n=== 模型性能对比 ===")
        print(res_df)
    return res_df

# 3.可视化：按用户活跃度分桶看RMSE
def plot_user_grp(data_dir, train_file):
    tr_df = pd.read_csv(train_file, header=None, names=['uid', 'iid', 'r'])
    u_counts = tr_df['uid'].value_counts()
    
    all_files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    files = [f for f in all_files if "_test.csv" not in f and "layers[" not in f]
    plot_data = []
    
    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = pd.read_csv(f, header=None, names=['uid', 'iid', 'yt', 'yp'])
        df['cnt'] = df['uid'].map(u_counts).fillna(0)
        
        def get_grp(c):
            if c <= 20: return "冷启动(≤20)"
            elif c <= 100: return "中活跃(21-100)"
            else: return "高活跃(>100)"
            
        df['grp'] = df['cnt'].apply(get_grp)
        
        for grp_name, sub_df in df.groupby('grp', observed=False):
            _, rmse = cal_metrics(sub_df['yt'].values, sub_df['yp'].values)
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

# 4.可视化：按电影热度分桶看RMSE
def plot_item_grp(data_dir, train_file):
    tr_df = pd.read_csv(train_file, header=None, names=['uid', 'iid', 'r'])
    i_counts = tr_df['iid'].value_counts()
    
    all_files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    files = [f for f in all_files if "_test.csv" not in f and "layers[" not in f]
    plot_data = []
    
    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = pd.read_csv(f, header=None, names=['uid', 'iid', 'yt', 'yp'])
        df['cnt'] = df['iid'].map(i_counts).fillna(0)
        
        def get_item_grp(c):
            if c <= 10: return "冷门电影"
            elif c <= 50: return "普通电影"
            else: return "热门电影"
            
        df['grp'] = df['cnt'].apply(get_item_grp)
        
        for grp_name, sub_df in df.groupby('grp', observed=False):
            _, rmse = cal_metrics(sub_df['yt'].values, sub_df['yp'].values)
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

# 5.可视化：残差分布图
def plot_err_dist(data_dir, test_path='data/processed/test.csv'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 读取真实评分
    test_df = pd.read_csv(test_path)
    true_ratings = test_df['rating'].values

    # 暖色系基线（左）
    left_models = [
        {'file': 'pred_global_avg.csv', 'label': 'Global Mean',    'color': '#F4A582', 'lw': 1.0},
        {'file': 'pred_user_avg.csv',   'label': 'User Average',   'color': '#E8833A', 'lw': 1.2},
        {'file': 'pred_item_knn.csv',   'label': 'Item-KNN',       'color': '#CA0020', 'lw': 1.5},
    ]
    # 冷色系高级模型（右）
    right_models = [
        {'file': 'pred_mf_dim64.csv',   'label': 'MF (dim=64)',     'color': '#0571B0', 'lw': 2.0},
        {'file': 'pred_ncf_layer3.csv', 'label': 'NCF [128,64,32]', 'color': '#008837', 'lw': 2.0},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    # 注意：不共享y轴，避免基线曲线被压缩

    # ----- 左图：基线模型 -----
    ax = axes[0]
    for mdl in left_models:
        f = os.path.join(data_dir, mdl['file'])
        if not os.path.exists(f):
            print(f"⚠️ 文件缺失: {mdl['file']}")
            continue
        df = pd.read_csv(f, header=None)
        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            yt = true_ratings
            yp = df['yp'].values
        else:  # 四列
            df.columns = ['uid', 'iid', 'yt', 'yp']
            yt = df['yt'].values
            yp = df['yp'].values
        err = yt - yp
        sns.histplot(err, kde=True, label=mdl['label'], stat="density",
                     bins=40, color=mdl['color'], alpha=0.35, linewidth=mdl['lw'], ax=ax)
    ax.axvline(x=0, color='#333333', linestyle='--', linewidth=1.2)
    ax.set_title('Baseline Models', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prediction Error', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(axis='y', alpha=0.2)

    # ----- 右图：高级模型 -----
    ax = axes[1]
    for mdl in right_models:
        f = os.path.join(data_dir, mdl['file'])
        if not os.path.exists(f):
            print(f"⚠️ 文件缺失: {mdl['file']}")
            continue
        df = pd.read_csv(f, header=None)
        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            yt = true_ratings
            yp = df['yp'].values
        else:
            df.columns = ['uid', 'iid', 'yt', 'yp']
            yt = df['yt'].values
            yp = df['yp'].values
        err = yt - yp
        sns.histplot(err, kde=True, label=mdl['label'], stat="density",
                     bins=40, color=mdl['color'], alpha=0.4, linewidth=mdl['lw'], ax=ax)
    ax.axvline(x=0, color='#333333', linestyle='--', linewidth=1.2)
    ax.set_title('Advanced Models', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prediction Error', fontsize=11)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(axis='y', alpha=0.2)

    plt.suptitle('Prediction Error Distribution Comparison', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = os.path.join(data_dir, 'err_dist_compare.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 图片已保存至 {output_path}")

# 6.MF模型不同维度对RMSE的影响趋势图
def plot_mf_trend(data_dir):
    files = glob.glob(os.path.join(data_dir, "pred_mf_dim*.csv"))
    files = [f for f in files if "_test.csv" not in f]
    if not files: return
    
    res = []
    for f in files:
        try:
            dim_str = os.path.basename(f).replace("pred_mf_dim", "").replace(".csv", "")
            dim = int(dim_str)
            df = pd.read_csv(f, header=None, names=['uid', 'iid', 'yt', 'yp'])
            _, rmse = cal_metrics(df['yt'].values, df['yp'].values)
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

# 7.数据稀疏性专题分析评估
def calc_sparse_gap(data_dir, train_file):
    tr_df = pd.read_csv(train_file, header=None, names=['uid', 'iid', 'r'])
    u_counts = tr_df['uid'].value_counts()
    
    all_files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    files = [f for f in all_files if "_test.csv" not in f and "layers[" not in f]
    gap_data = []
    
    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = pd.read_csv(f, header=None, names=['uid', 'iid', 'yt', 'yp'])
        df['cnt'] = df['uid'].map(u_counts).fillna(0)
        
        _, rmse_all = cal_metrics(df['yt'].values, df['yp'].values)
        df_cold = df[df['cnt'] <= 20]
        
        if len(df_cold) > 0 and rmse_all > 0:
            _, rmse_cold = cal_metrics(df_cold['yt'].values, df_cold['yp'].values)
            up_ratio = (rmse_cold - rmse_all) / rmse_all * 100
            gap_data.append({
                "Model": m_name, 
                "整体 RMSE": round(rmse_all, 4), 
                "冷启动 RMSE": round(rmse_cold, 4), 
                "恶化比例(%)": round(up_ratio, 2)
            })
        
    if gap_data:
        gap_df = pd.DataFrame(gap_data).sort_values("整体 RMSE")
        print("\n=== 数据稀疏性专题分析 ===")
        print(gap_df)
        gap_df.to_csv(os.path.join(data_dir, "sparsity_analysis.csv"), index=False)

if __name__ == "__main__":
    out_dir = "outputs"
    train_path = "data/processed/train.csv"
    
    print("开始运行全模型完整指标评估")
    eval_all(out_dir)
    plot_user_grp(out_dir, train_path)
    plot_item_grp(out_dir, train_path)
    plot_err_dist(out_dir)
    plot_mf_trend(out_dir)
    calc_sparse_gap(out_dir, train_path)
    print("\n所有可视化图表与数据分析已成功生成在outputs/目录下！")
