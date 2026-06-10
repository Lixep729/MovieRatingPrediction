import os
import glob
import numpy as np
import pandas as pd

def describe_errors(data_dir="outputs", test_path="data/processed/test.csv"):
    test_df = pd.read_csv(test_path)
    true_all = test_df['rating'].values

    files = glob.glob(os.path.join(data_dir, "pred_*.csv"))
    # 只分析我们要画的几个代表模型
    target = {
        "pred_global_avg.csv": "Global Mean",
        "pred_user_avg.csv":   "User Average",
        "pred_item_knn.csv":   "Item-KNN",
        "pred_mf_dim64.csv":   "MF (dim=64)",
        "pred_ncf_layer3.csv": "NCF [128,64,32]",
    }

    for fname, label in target.items():
        f = os.path.join(data_dir, fname)
        if not os.path.exists(f):
            print(f"\n{label}: 文件不存在")
            continue
        df = pd.read_csv(f, header=None)
        if df.shape[1] == 3:
            df.columns = ['uid', 'iid', 'yp']
            yt = true_all
            yp = df['yp'].values
        else:
            df.columns = ['uid', 'iid', 'yt', 'yp']
            yt = df['yt'].values
            yp = df['yp'].values

        err = yt - yp
        print(f"\n{'='*50}")
        print(f"模型：{label}")
        print(f"样本数：{len(err)}")
        print(f"误差均值：{np.mean(err):.4f}")
        print(f"误差标准差：{np.std(err):.4f}")
        print(f"误差最小值：{np.min(err):.4f}")
        print(f"误差最大值：{np.max(err):.4f}")
        print(f"偏度：{pd.Series(err).skew():.4f}")
        print(f"峰度：{pd.Series(err).kurtosis():.4f}")
        # 误差落在常用区间的比例
        for rng, desc in [([-0.5, 0.5], "±0.5"),
                          ([-1.0, 1.0], "±1.0"),
                          ([-2.0, 2.0], "±2.0")]:
            prop = np.mean((err >= rng[0]) & (err <= rng[1])) * 100
            print(f"误差在{desc}内的比例：{prop:.1f}%")

if __name__ == "__main__":
    describe_errors()