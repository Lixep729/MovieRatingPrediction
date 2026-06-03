import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot') 

BAR_COLOR = '#4C72B0'      
HIST_COLOR_1 = '#E67E22'  
HIST_COLOR_2 = '#27AE60'   
EDGE_COLOR = '#2C3E50'    
GRID_COLOR = '#EAEAF2'    

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': GRID_COLOR,
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

def main():
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')
    df = pd.concat([train, test], ignore_index=True)
    print(f"训练集: {len(train)} 条  测试集: {len(test)} 条  总计: {len(df)} 条")

    n_users = df['user_id'].max()
    n_items = df['item_id'].max()
    n_ratings = len(df)
    sparsity = 1 - n_ratings / (n_users * n_items)

    print("\n========== 数据统计 ==========")
    print(f"用户数: {n_users}")
    print(f"电影数: {n_items}")
    print(f"评分数: {n_ratings}")
    print(f"稀疏度: {sparsity:.4f} ({sparsity*100:.2f}%)")

    summary = (
        f"本次使用的 MovieLens 100K 数据集包含 {n_users} 名用户，"
        f"{n_items} 部电影，共计 {n_ratings} 条评分记录，"
        f"评分矩阵稀疏度为 {sparsity:.4f}（{sparsity*100:.2f}%）。"
    )
    print("\n--- 报告文字素材 ---")
    print(summary)

    os.makedirs('outputs', exist_ok=True)

    #图1：评分分布
    rating_counts = df['rating'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(rating_counts.index, rating_counts.values,
                   color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.8, width=0.6)
    for bar, val in zip(bars, rating_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
                 f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Distribution of Ratings', fontweight='bold', pad=15)
    plt.xlabel('Rating')
    plt.ylabel('Number of Ratings')
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(0, rating_counts.max() * 1.15) 
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('outputs/rating_distribution.png')
    plt.close()
    print("已保存 outputs/rating_distribution.png")

    #图2：用户评分数分布
    user_counts = df['user_id'].value_counts()
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(user_counts, bins=50,
                                color=HIST_COLOR_1, edgecolor='white', alpha=0.85)
    plt.title('Distribution of Ratings per User', fontweight='bold', pad=15)
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.grid(axis='y', alpha=0.3)
    mean_val = user_counts.mean()
    plt.axvline(mean_val, color='black', linestyle='--', linewidth=1.2,
                label=f'Mean: {mean_val:.1f}')
    plt.legend()
    plt.savefig('outputs/user_rating_count.png')
    plt.close()
    print("已保存 outputs/user_rating_count.png")

    #图3：电影被评次数分布
    item_counts = df['item_id'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.hist(item_counts, bins=50, color=HIST_COLOR_2, edgecolor='white', alpha=0.85)
    plt.title('Distribution of Ratings per Movie', fontweight='bold', pad=15)
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Movies')
    plt.grid(axis='y', alpha=0.3)
    mean_item = item_counts.mean()
    plt.axvline(mean_item, color='black', linestyle='--', linewidth=1.2,
                label=f'Mean: {mean_item:.1f}')
    plt.legend()
    plt.savefig('outputs/item_rating_count.png')
    plt.close()
    print("已保存 outputs/item_rating_count.png")


if __name__ == '__main__':
    main()
