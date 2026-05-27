import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mindspore as ms
import mindspore.ops as ops

# 基本画图设置,防止中文乱码
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

# 5.可视化：残差分布图，精选代表性模型避免线条过密
def plot_err_dist(data_dir):
    target_models = ['global_avg', 'user_avg', 'item_knn', 'mf_dim32', 'ncf_layer3']
    plt.figure(figsize=(8, 4.5))
    
    has_data = False
    for m_name in target_models:
        f = os.path.join(data_dir, f"pred_{m_name}.csv")
        if os.path.exists(f):
            df = pd.read_csv(f, header=None, names=['uid', 'iid', 'yt', 'yp'])
            if len(df) > 0:
                err = df['yt'] - df['yp']
                sns.histplot(err, kde=True, label=m_name, stat="count", alpha=0.2, bins=30)
                has_data = True
        
    if has_data:
        plt.axvline(x=0, color='r', linestyle='--', label='Perfect')
        plt.title("核心模型预测偏差（残差）分布图")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "err_dist.png"))
    plt.close()

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

#主运行流程
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