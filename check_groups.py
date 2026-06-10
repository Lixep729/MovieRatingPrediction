import os
import glob
import numpy as np
import pandas as pd

def print_group_errors(data_dir="outputs", train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
    # 读取训练集
    tr_df = pd.read_csv(train_path)
    tr_df.columns = ['uid', 'iid', 'r']
    u_counts = tr_df['uid'].value_counts()
    i_counts = tr_df['iid'].value_counts()
    
    # 读取测试集
    test_df = pd.read_csv(test_path)
    true_all = test_df['rating'].values
    
    files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    files = [f for f in files if "_test.csv" not in f and "layers[" not in f]
    
    # 按用户活跃度分组
    print("=" * 80)
    print("按用户活跃度分组 RMSE")
    print("=" * 80)
    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = pd.read_csv(f, header=None)
        
        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            df['yt'] = true_all
        else:
            df.columns = ['uid', 'iid', 'yt', 'yp']
        
        df['uid'] = df['uid'].astype(int)
        df['cnt'] = df['uid'].map(u_counts).fillna(0).astype(int)
        
        def get_grp(c):
            if c <= 20: return "冷启动(≤20)"
            elif c <= 100: return "中活跃(21-100)"
            else: return "高活跃(>100)"
        df['grp'] = df['cnt'].apply(get_grp)
        
        print(f"\n{m_name}:")
        for grp_name in ["冷启动(≤20)", "中活跃(21-100)", "高活跃(>100)"]:
            sub = df[df['grp'] == grp_name]
            if len(sub) > 0:
                rmse = np.sqrt(np.mean((sub['yt'].values - sub['yp'].values) ** 2))
                print(f"  {grp_name}: 样本数={len(sub)}, RMSE={rmse:.4f}")
    
    # 按电影热度分组
    print("\n" + "=" * 80)
    print("按电影热度分组 RMSE")
    print("=" * 80)
    for f in files:
        m_name = os.path.basename(f).replace("pred_", "").replace(".csv", "")
        df = pd.read_csv(f, header=None)
        
        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            df['yt'] = true_all
        else:
            df.columns = ['uid', 'iid', 'yt', 'yp']
        
        df['iid'] = df['iid'].astype(int)
        df['cnt'] = df['iid'].map(i_counts).fillna(0).astype(int)
        
        def get_item_grp(c):
            if c <= 10: return "冷门(≤10)"
            elif c <= 50: return "普通(11-50)"
            else: return "热门(>50)"
        df['grp'] = df['cnt'].apply(get_item_grp)
        
        print(f"\n{m_name}:")
        for grp_name in ["冷门(≤10)", "普通(11-50)", "热门(>50)"]:
            sub = df[df['grp'] == grp_name]
            if len(sub) > 0:
                rmse = np.sqrt(np.mean((sub['yt'].values - sub['yp'].values) ** 2))
                print(f"  {grp_name}: 样本数={len(sub)}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    print_group_errors()