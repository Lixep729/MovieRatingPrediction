import matplotlib.pyplot as plt
import numpy as np

dims = [8, 16, 32, 64]
train_time = [25.62, 26.92, 31.73, 33.25]
final_loss = [0.750520, 0.659808, 0.543610, 0.351811]
rmse = [0.9157, 0.9152, 0.9130, 0.9318]
mae = [0.7220, 0.7182, 0.7174, 0.7316]

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel("Embedding Dimension")
ax1.set_ylabel("Loss / Error", color="#d62728")
ax1.plot(dims, rmse, marker="o", label="RMSE", color="#1f77b4", linewidth=2)
ax1.plot(dims, mae, marker="s", label="MAE", color="#ff7f0e", linewidth=2)
ax1.plot(dims, final_loss, marker="^", label="Final Train Loss", color="#2ca02c", linewidth=2)
ax1.tick_params(axis="y", labelcolor="#d62728")

ax2 = ax1.twinx()
ax2.set_ylabel("Training Time (s)", color="#9467bd")
ax2.plot(dims, train_time, marker="D", label="Train Time", color="#9467bd", linewidth=2)
ax2.tick_params(axis="y", labelcolor="#9467bd")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.title("MF Model Hyperparameter Comparison (Embedding Dimension)")
plt.grid(alpha=0.3)
plt.savefig("outputs/mf_embedding_dim_comparison.png", dpi=300, bbox_inches="tight")
plt.show()